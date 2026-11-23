"""Lemonade Realtime WebSocket provider for the standalone voice lab.

A batch text-to-speech adapter that drives the local Lemonade realtime
bridge running on the local host (default ``ws://127.0.0.1:8010/v1/realtime``)
with the OpenAI Realtime wire protocol, then records the resulting PCM
audio deltas into a WAV. This is the lab-side analogue of
``relay/realtime_agent/providers/lemonade.py``: the relay broker uses the
async version for live sessions; this file is the synchronous one-shot
CLI/benchmark tool.

The Lemonade bridge speaks OpenAI Realtime verbatim. No auth header is
required — the bridge runs on the same host as the relay and accepts a
stub ``Authorization: Bearer local`` header. We set ``turn_detection: null``
and disable transcription so we don't double-pay for STT on a pure TTS
task.
"""

from __future__ import annotations

import base64
import json
import os
import wave
from collections.abc import Callable
from typing import Any, Protocol

from ..metrics import MetricsRecorder
from .base import (
    ProviderInfo,
    ProviderRunError,
    ProviderUnavailable,
    VoiceProvider,
    VoiceRequest,
    VoiceResponse,
)

DEFAULT_MODEL = "Bonsai-1.7B-gguf"
DEFAULT_VOICE = "af_heart"
DEFAULT_OUTPUT_FORMAT = "audio/pcm"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_URL = "ws://127.0.0.1:8010/v1/realtime"
DEFAULT_AUTH_TOKEN = "local"

_AUDIO_DELTA_EVENTS = {
    "response.audio.delta",
    "response.output_audio.delta",
}
_AUDIO_DONE_EVENTS = {
    "response.audio.done",
    "response.output_audio.done",
}
_TEXT_DELTA_EVENTS = {
    "response.output_audio_transcript.delta",
    "response.audio_transcript.delta",
    "response.output_text.delta",
    "response.text.delta",
}


class WebSocketLike(Protocol):
    def send(self, payload: str) -> Any: ...

    def recv(self) -> str | bytes: ...

    def close(self) -> Any: ...


SocketFactory = Callable[[str, list[str], float], WebSocketLike]


class LemonadeRealtimeProvider(VoiceProvider):
    """Render text prompts through the local Lemonade realtime bridge."""

    info = ProviderInfo(
        id="lemonade_realtime",
        name="Lemonade Realtime (Local Bridge)",
        description=(
            "Server-side WebSocket adapter for the local Lemonade realtime "
            "bridge running on the local host. Pushes text through the bridge's "
            "OpenAI-Realtime-compatible /v1/realtime endpoint, which routes "
            "to Lemonade Whisper-Large-v3-Turbo STT + Lemonade LLM (Bonsai "
            "or 35B) + Lemonade Kokoro TTS. The live voice-agent path lives "
            "in relay/realtime_agent/providers/lemonade.py."
        ),
        models=("Bonsai-1.7B-gguf",),
        voices=(
            "af_heart",
            "af_bella",
            "af_nicole",
            "am_adam",
            "bm_fable",
            "Watercooler",
        ),
        sample_rates=(24000,),
        supports_tts=True,
        supports_realtime=True,
        supports_interruption=False,
        supports_expression=True,
    )

    def __init__(self, socket_factory: SocketFactory | None = None) -> None:
        self._socket_factory = socket_factory or _create_websocket

    def run_text(
        self,
        request: VoiceRequest,
        recorder: MetricsRecorder,
    ) -> VoiceResponse:
        options = request.provider_options
        url_base = (
            _option(options, "url")
            or os.getenv("LEMONADE_REALTIME_URL")
            or DEFAULT_URL
        ).rstrip("/")
        model = _option(options, "model") or DEFAULT_MODEL
        voice = _option(options, "voice") or DEFAULT_VOICE
        sample_rate = _int_option(options, "sample_rate", DEFAULT_SAMPLE_RATE)
        timeout = _float_option(options, "timeout", DEFAULT_TIMEOUT_SECONDS)
        auth_token = _option(options, "api_key") or DEFAULT_AUTH_TOKEN

        request.output_path.parent.mkdir(parents=True, exist_ok=True)

        headers = [f"Authorization: Bearer {auth_token}"]

        recorder.event(
            "request_started",
            provider=self.info.id,
            model=model,
            voice=voice,
            text_chars=len(request.text),
            expression=request.expression.to_dict(),
        )

        ws: WebSocketLike | None = None
        transcript_parts: list[str] = []
        audio_bytes = 0
        events_seen: list[str] = []
        try:
            ws = self._socket_factory(url_base, headers, timeout)
            recorder.event("websocket_connected", provider=self.info.id, model=model)
            _send_json(
                ws,
                _session_update(
                    model=model,
                    voice=voice,
                    sample_rate=sample_rate,
                ),
            )
            recorder.event("client_event_sent", client_event_type="session.update")
            _send_json(ws, _conversation_item_create(request))
            recorder.event(
                "client_event_sent",
                client_event_type="conversation.item.create",
            )
            _send_json(ws, _response_create())
            recorder.event("client_event_sent", client_event_type="response.create")

            with wave.open(str(request.output_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)

                while True:
                    event = _recv_json(ws)
                    event_type = str(event.get("type", "unknown"))
                    events_seen.append(event_type)
                    recorder.event(
                        "server_event",
                        server_event_type=event_type,
                        response_id=event.get("response_id"),
                        item_id=event.get("item_id"),
                    )

                    if event_type == "error":
                        raise ProviderRunError(_format_lemonade_error(event))

                    if event_type in _TEXT_DELTA_EVENTS:
                        delta = event.get("delta")
                        if isinstance(delta, str):
                            transcript_parts.append(delta)
                        continue

                    if event_type in _AUDIO_DELTA_EVENTS:
                        chunk = _decode_audio_delta(event)
                        wav.writeframes(chunk)
                        audio_bytes += len(chunk)
                        recorder.audio_chunk(
                            byte_count=len(chunk),
                            label=event_type,
                            pcm=chunk,
                        )
                        if request.audio_sink is not None:
                            request.audio_sink(
                                chunk,
                                {
                                    "label": event_type,
                                    "sample_rate": sample_rate,
                                    "channels": 1,
                                    "sample_width": 2,
                                },
                            )
                        continue

                    if event_type in _AUDIO_DONE_EVENTS:
                        continue

                    if event_type == "response.done":
                        break
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

        if audio_bytes <= 0:
            _delete_empty_output(request.output_path)
            raise ProviderRunError(
                "Lemonade Realtime completed without audio delta events"
            )

        metrics = recorder.finish(provider=self.info.id, model=model, voice=voice)
        return VoiceResponse(
            provider=self.info.id,
            model=model,
            voice=voice,
            audio_path=request.output_path,
            metrics=metrics,
            metadata={
                "audio_bytes": audio_bytes,
                "output_format": "audio/pcm",
                "sample_rate": sample_rate,
                "server_event_types": events_seen,
                "transcript": "".join(transcript_parts).strip() or None,
            },
        )


def _create_websocket(url: str, headers: list[str], timeout: float) -> WebSocketLike:
    try:
        import websocket  # type: ignore
    except ImportError as exc:
        raise ProviderUnavailable(
            "lemonade_realtime requires websocket-client; install with "
            "`pip install websocket-client`"
        ) from exc
    return websocket.create_connection(url, header=headers, timeout=timeout)


def _session_update(*, model: str, voice: str, sample_rate: int) -> dict[str, Any]:
    # Disable input transcription and VAD — the lab is rendering text to
    # speech, not transcribing audio.
    return {
        "type": "session.update",
        "session": {
            "model": model,
            "voice": voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": None,
        },
    }


def _conversation_item_create(request: VoiceRequest) -> dict[str, Any]:
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": request.text}],
        },
    }


def _response_create() -> dict[str, Any]:
    return {"type": "response.create"}


def _decode_audio_delta(event: dict[str, Any]) -> bytes:
    payload = event.get("delta") or event.get("audio")
    if not isinstance(payload, str):
        raise ProviderRunError("Lemonade Realtime audio delta missing base64 payload")
    try:
        return base64.b64decode(payload)
    except Exception as exc:
        raise ProviderRunError(
            "Lemonade Realtime audio delta was not valid base64"
        ) from exc


def _format_lemonade_error(event: dict[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error.get("type")
        if message:
            return f"Lemonade Realtime error: {message}"
    return f"Lemonade Realtime error event: {event}"


def _option(options: dict[str, Any], name: str) -> str | None:
    value = options.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_option(options: dict[str, Any], name: str, default: int) -> int:
    value = _option(options, name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ProviderUnavailable(f"{name} must be an integer") from exc


def _float_option(options: dict[str, Any], name: str, default: float) -> float:
    value = _option(options, name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ProviderUnavailable(f"{name} must be a number") from exc


def _send_json(ws: WebSocketLike, payload: dict[str, Any]) -> None:
    ws.send(json.dumps(payload, separators=(",", ":")))


def _recv_json(ws: WebSocketLike) -> dict[str, Any]:
    raw = ws.recv()
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProviderRunError(
                "Lemonade Realtime returned non-UTF-8 binary frame"
            ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderRunError(
            "Lemonade Realtime returned non-JSON text"
        ) from exc
    if not isinstance(payload, dict):
        raise ProviderRunError(
            "Lemonade Realtime returned a non-object event"
        )
    return payload


def _delete_empty_output(path: Any) -> None:
    try:
        if hasattr(path, "exists") and path.exists():
            path.unlink()
    except OSError:
        pass

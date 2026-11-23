"""Unmute Realtime WebSocket provider for the standalone voice lab.

A batch text-to-speech adapter that drives the Kyutai unmute backend's
``/v1/realtime`` endpoint with the OpenAI Realtime wire protocol, then
records the resulting PCM audio deltas into a WAV. This is the lab-side
analogue of ``relay/realtime_agent/providers/unmute.py``: the relay
broker uses the async version for live sessions; this file is the
synchronous one-shot CLI/benchmark tool.

The unmute backend speaks the OpenAI Realtime protocol verbatim, with
the only difference being the auth header (``kyutai-api-key`` instead of
``Authorization: Bearer``). For batch renders from the lab, we set
``turn_detection: null`` and disable transcription so we don't double-pay
for STT on a pure TTS task.
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

DEFAULT_MODEL = "moshi-realtime"
DEFAULT_VOICE = "Watercooler"
DEFAULT_OUTPUT_FORMAT = "audio/pcm"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_URL = "ws://localhost:8000/v1/realtime"
DEFAULT_PUBLIC_TOKEN = "public_token"

_AUDIO_DELTA_EVENTS = {
    "response.audio.delta",
    "response.output_audio.delta",
}
_AUDIO_DONE_EVENTS = {
    "response.audio.done",
    "response.output_audio.done",
}
_TEXT_DELTA_EVENTS = {
    "response.output_text.delta",
    "response.text.delta",
    "response.audio_transcript.delta",
}


class WebSocketLike(Protocol):
    def send(self, payload: str) -> Any: ...

    def recv(self) -> str | bytes: ...

    def close(self) -> Any: ...


SocketFactory = Callable[[str, list[str], float], WebSocketLike]


class UnmuteRealtimeProvider(VoiceProvider):
    """Render text prompts through the Kyutai unmute backend over WebSocket."""

    info = ProviderInfo(
        id="unmute_realtime",
        name="Kyutai Unmute (Realtime)",
        description=(
            "Server-side WebSocket adapter for the Kyutai Unmute backend's "
            "OpenAI-Realtime-compatible /v1/realtime endpoint. Drives Moshi MLX "
            "STT/TTS on a local Mac (or any host running unmunte-backend) with "
            "Hermes as the LLM. For batch text-to-speech rendering "
            "only — the live voice-agent path lives in "
            "relay/realtime_agent/providers/unmute.py."
        ),
        models=(DEFAULT_MODEL,),
        voices=("Watercooler", "Cartesia-British-Reading-Lady"),
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
        api_key = _resolve_api_key(options)
        url_base = (_option(options, "url") or os.getenv("UNMUTE_REALTIME_URL") or DEFAULT_URL).rstrip("/")
        model = _option(options, "model") or DEFAULT_MODEL
        voice = _option(options, "voice") or DEFAULT_VOICE
        output_format = _option(options, "output_format") or DEFAULT_OUTPUT_FORMAT
        if output_format != DEFAULT_OUTPUT_FORMAT:
            raise ProviderUnavailable(
                "unmute_realtime currently supports output_format=audio/pcm only"
            )
        sample_rate = _int_option(options, "sample_rate", DEFAULT_SAMPLE_RATE)
        timeout = _float_option(options, "timeout", DEFAULT_TIMEOUT_SECONDS)

        request.output_path.parent.mkdir(parents=True, exist_ok=True)

        # The orchestrator defaults to a public ``kyutai-api-key: public_token``
        # scheme; operators can override via env or provider_options. We do NOT
        # accept a bearer token from the CLI — the relay broker handles that
        # surface for live sessions.
        headers = [f"kyutai-api-key: {api_key}"]

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
                        raise ProviderRunError(_format_unmute_error(event))

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
                "Unmute Realtime completed without audio delta events"
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
                "output_format": output_format,
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
            "unmute_realtime requires websocket-client; install with "
            "`pip install websocket-client`"
        ) from exc
    return websocket.create_connection(url, header=headers, timeout=timeout)


def _session_update(*, voice: str, sample_rate: int) -> dict[str, Any]:
    # Disable input transcription — the lab is rendering text to speech,
    # not transcribing audio. Disable server VAD — there's no microphone.
    return {
        "type": "session.update",
        "session": {
            "voice": voice,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": sample_rate},
                    "turn_detection": None,
                    "transcription": None,
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": sample_rate},
                    "voice": voice,
                },
            },
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
        raise ProviderRunError("Unmute Realtime audio delta missing base64 payload")
    try:
        return base64.b64decode(payload)
    except Exception as exc:
        raise ProviderRunError(
            "Unmute Realtime audio delta was not valid base64"
        ) from exc


def _format_unmute_error(event: dict[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error.get("type")
        if message:
            return f"Unmute Realtime error: {message}"
    return f"Unmute Realtime error event: {event}"


def _resolve_api_key(options: dict[str, Any]) -> str:
    explicit = _option(options, "api_key")
    if explicit:
        return explicit
    env = os.getenv("UNMUTE_REALTIME_API_KEY") or os.getenv("UNMUTE_API_KEY")
    if env:
        return env.strip()
    return DEFAULT_PUBLIC_TOKEN


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
                "Unmute Realtime returned non-UTF-8 binary frame"
            ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderRunError("Unmute Realtime returned non-JSON text") from exc
    if not isinstance(payload, dict):
        raise ProviderRunError("Unmute Realtime returned a non-object event")
    return payload


def _delete_empty_output(path: Any) -> None:
    try:
        if hasattr(path, "exists") and path.exists():
            path.unlink()
    except OSError:
        pass

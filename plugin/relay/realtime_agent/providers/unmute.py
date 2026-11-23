"""Unmute provider-native realtime-agent connection.

The unmute backend (Kyutai's `unmute-backend`) speaks the OpenAI Realtime
WebSocket protocol verbatim at ``/v1/realtime``. We reuse the same
session shape, audio deltas, transcripts, and function-call event flow
as ``OpenAIRealtimeAgentProvider`` — only the URL, auth header, and a
small subset of session knobs differ (unmute prefers its own ``voice``
field over ``output_modalities`` + ``audio.output.voice``).

The orchestrator fans audio through its own moshi-mlx STT/TTS bridge and
forwards LLM requests to the configured Hermes API server. The relay
plugin's job is just to be a polite OpenAI Realtime client.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import urllib.parse
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp

from ....voice_lab.providers.base import ProviderRunError, ProviderUnavailable

from ..models import (
    ProviderEvent,
    ProviderEventKind,
    RealtimeAgentCapabilities,
    RealtimeAgentSessionConfig,
    ToolCallEvent,
)
from .base import RealtimeAgentProviderAdapter

DEFAULT_URL = "ws://localhost:8000/v1/realtime"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_PUBLIC_TOKEN = "public_token"
SUPPORTED_SAMPLE_RATE = 24000
# The unmute orchestrator decodes incoming ``input_audio_buffer.append``
# payloads via ``sphn.OpusStreamReader`` at a hard-coded 24 kHz mono
# rate (see unmute/kyutai_constants.py::SAMPLE_RATE). We must therefore
# pre-resample + Opus-encode every audio chunk before sending.
UNMUTE_OPUS_SAMPLE_RATE = 24000


class _OpusStreamWriterLazy:
    """Lazy wrapper around ``sphn.OpusStreamWriter``.

    Keeps ``sphn`` an optional dependency: the writer is only constructed
    when the first audio chunk arrives, so test environments without
    libopus installed (or with a stubbed orchestrator) still import this
    module cleanly. ``append_pcm_bytes`` is sync — the provider always
    runs it via ``asyncio.to_thread`` because Opus encoding is CPU-bound.
    """

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._writer: Any | None = None

    def _ensure_writer(self) -> Any:
        if self._writer is None:
            import numpy as np
            import sphn

            self._writer = sphn.OpusStreamWriter(self._sample_rate)
            self._np = np
        return self._writer

    def append_pcm_bytes(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""
        writer = self._ensure_writer()
        np = self._np
        samples = np.frombuffer(pcm, dtype=np.int16)
        # sphn wants float32 mono in [-1, 1]
        pcm_f = samples.astype(np.float32) / 32768.0
        return bytes(writer.append_pcm(pcm_f))

# Reuse the OpenAI Realtime event name sets verbatim — the orchestrator
# relays these unchanged from Moshi/Unmute's external-tag envelope.
_AUDIO_DELTA_EVENTS = {
    "response.output_audio.delta",
    "response.audio.delta",
}
_AUDIO_DONE_EVENTS = {
    "response.output_audio.done",
    "response.audio.done",
}
_INPUT_TRANSCRIPT_DELTA_EVENTS = {
    "conversation.item.input_audio_transcription.delta",
    "input_audio_transcription.delta",
}
_INPUT_TRANSCRIPT_FINAL_EVENTS = {
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.input_audio_transcription.done",
    "input_audio_transcription.completed",
    "input_audio_transcription.done",
}
_OUTPUT_TEXT_DELTA_EVENTS = {
    "response.output_audio_transcript.delta",
    "response.audio_transcript.delta",
    "response.output_text.delta",
    "response.text.delta",
}
_FUNCTION_ITEM_DONE_EVENTS = {
    "response.output_item.done",
    "conversation.item.done",
}


class UnmuteProviderSocket(Protocol):
    async def send_json(self, payload: dict[str, Any]) -> None: ...

    async def receive_json(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


SocketFactory = Callable[[str, dict[str, str], float], Any]


@dataclass(frozen=True, slots=True)
class UnmuteAuthToken:
    """Auth material for the unmute backend.

    The orchestrator defaults to a public ``kyutai-api-key: public_token``
    scheme but also accepts ``Authorization: Bearer ...`` against its
    own LLM-side bearer token if the operator wires one up. Both are
    passed through to the websocket handshake.
    """

    bearer: str | None
    api_key: str
    source: str


class UnmuteRealtimeAgentAdapter(RealtimeAgentProviderAdapter):
    """Render-only unmute adapter (kept for symmetry with xai/openai)."""

    provider_id = "unmute_realtime"


class UnmuteRealtimeAgentProvider:
    """Connection-oriented unmute (Kyutai Moshi) provider for Hermes Realtime Agent."""

    provider_id = "unmute_realtime"
    capabilities = RealtimeAgentCapabilities(
        provider_id=provider_id,
        supports_function_calls=True,
        supports_server_vad=True,
        input_audio_format="pcm16",
        output_audio_format="pcm16",
        accepts_text_input=True,
    )

    def __init__(self, socket_factory: SocketFactory | None = None) -> None:
        self._socket_factory = socket_factory or _create_aiohttp_websocket

    async def connect(
        self,
        config: RealtimeAgentSessionConfig,
    ) -> "UnmuteRealtimeAgentConnection":
        if config.sample_rate != SUPPORTED_SAMPLE_RATE:
            raise ProviderUnavailable(
                "Unmute realtime currently supports sample_rate=24000 for "
                "provider-native Realtime Agent sessions."
            )
        auth = _resolve_auth_token(config.provider_options)
        timeout = _float_option(
            config.provider_options, "timeout", DEFAULT_TIMEOUT_SECONDS
        )
        base_url = (
            _option(config.provider_options, "url")
            or os.getenv("UNMUTE_REALTIME_URL")
            or DEFAULT_URL
        ).rstrip("/")
        # unmute ignores the `?model=` query parameter the OpenAI Realtime
        # protocol uses for routing — it picks a model from the orchestrator's
        # own catalog. We still allow an explicit override for tests.
        url = base_url
        explicit_model = _option(config.provider_options, "model")
        if explicit_model:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}model={urllib.parse.quote(explicit_model, safe='')}"

        headers: dict[str, str] = {"kyutai-api-key": auth.api_key}
        if auth.bearer:
            headers["Authorization"] = f"Bearer {auth.bearer}"

        try:
            socket = await self._socket_factory(url, headers, timeout)
        except aiohttp.WSServerHandshakeError as exc:
            raise ProviderUnavailable(
                f"Unmute realtime backend at {url} rejected the handshake "
                f"(HTTP {exc.status}; auth source: {auth.source}). Confirm "
                "the orchestrator is running and reachable from the gateway."
            ) from exc
        connection = UnmuteRealtimeAgentConnection(
            socket=socket,
            config=config,
            auth_source=auth.source,
        )
        await connection.configure()
        return connection


class UnmuteRealtimeAgentConnection:
    def __init__(
        self,
        *,
        socket: UnmuteProviderSocket,
        config: RealtimeAgentSessionConfig,
        auth_source: str,
    ) -> None:
        self.socket = socket
        self.config = config
        self.auth_source = auth_source
        self._completed_function_call_ids: set[str] = set()
        # The unmute orchestrator's ``receive_loop`` runs every incoming
        # ``input_audio_buffer.append`` payload through
        # ``sphn.OpusStreamReader`` (Opus-in-OGG framing). PCM bytes
        # throw ``ValueError: unexpected ogg capture pattern`` on the
        # first chunk and tear the session down. So we hold a single
        # OpusStreamWriter keyed to the orchestrator's canonical
        # 24 kHz mono STT rate and pre-resample any phone-format PCM
        # to that rate before encoding.
        self._opus_writer = _OpusStreamWriterLazy(UNMUTE_OPUS_SAMPLE_RATE)

    async def configure(self) -> None:
        await self.socket.send_json(_session_update(self.config))

    async def send_audio(self, pcm: bytes, sample_rate: int) -> None:
        pcm = _resample_pcm16_mono(pcm, sample_rate, UNMUTE_OPUS_SAMPLE_RATE)
        opus_bytes = await asyncio.to_thread(
            self._opus_writer.append_pcm_bytes, pcm
        )
        if not opus_bytes:
            # Opus buffers internally; a tiny frame can produce no bytes
            # yet. Don't send an empty audio chunk — the orchestrator
            # would treat it as a malformed OGG page and close the WS.
            return
        await self.socket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(opus_bytes).decode("ascii"),
            }
        )

    async def commit_audio(self) -> None:
        # The unmute orchestrator's pydantic ClientEvent union only accepts
        # session.update, input_audio_buffer.append, and
        # unmute.input_audio_buffer.append_anonymized — anything else raises
        # an "Invalid message" validation error mid-session. The unmute
        # backend streams audio into STT continuously (no commit boundary),
        # so this is a no-op on the unmute provider. See
        # `openai_realtime_api_events.ClientEvent`.
        return

    async def send_text(self, text: str) -> None:
        # The unmute orchestrator only accepts three client tags
        # (session.update / input_audio_buffer.append /
        # unmute.input_audio_buffer.append_anonymized). Sending
        # conversation.item.create or response.create raises
        # "Invalid message" — both are ignored. Real text injection on the
        # unmute backend requires a patched orchestrator; callers should
        # use a session.update payload instead.
        return

    async def clear_audio(self) -> None:
        # See commit_audio() — the unmute orchestrator rejects
        # input_audio_buffer.clear with "Invalid message" because it isn't in
        # the ClientEvent union. Unmute has no equivalent; audio is a live
        # stream that the orchestrator/handler controls internally.
        return

    async def cancel_response(self) -> None:
        # See commit_audio() — response.cancel isn't in the ClientEvent union
        # and would be rejected as "Invalid message". The unmute orchestrator
        # emits its own UnmuteInterruptedByVAD when STT detects barge-in,
        # so cancellation isn't needed at this layer.
        return

    async def send_tool_result(self, call_id: str, output: dict[str, Any]) -> None:
        # conversation.item.create isn't accepted by the unmute orchestrator's
        # ClientEvent union. Tool result delivery needs server-side support;
        # callers should fall back to including tool output in the next
        # session.update or in audio context.
        return

    async def append_context_item(self, *, role: str, text: str) -> None:
        # conversation.item.create isn't in the unmute ClientEvent union; the
        # orchestrator would reject it as "Invalid message". Unmute has no
        # explicit "history item" primitive on this WS — context is fed
        # implicitly through audio and the session.update instructions.
        return

    async def request_response(
        self,
        *,
        instructions: str | None = None,
        exact_text: str | None = None,
    ) -> None:
        # response.create isn't accepted by the unmute orchestrator. On unmute
        # the backend's own VAD + LLM pipeline decides when to start a
        # response; there's no client-initiated trigger. No-op here.
        return

    async def close(self) -> None:
        await self.socket.close()

    async def events(self) -> AsyncIterator[ProviderEvent]:
        while True:
            try:
                raw = await self.socket.receive_json()
            except (EOFError, StopAsyncIteration):
                return
            except ProviderRunError as exc:
                yield ProviderEvent(
                    ProviderEventKind.ERROR,
                    payload={"message": str(exc)},
                )
                return
            event = _provider_event(raw, self._completed_function_call_ids)
            if event is not None:
                yield event


class _AioHttpUnmuteProviderSocket:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
    ) -> None:
        self._session = session
        self._ws = ws

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._ws.send_str(json.dumps(payload, separators=(",", ":")))

    async def receive_json(self) -> dict[str, Any]:
        msg = await self._ws.receive()
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError as exc:
                raise ProviderRunError(
                    "Unmute realtime returned non-JSON text"
                ) from exc
            if not isinstance(payload, dict):
                raise ProviderRunError(
                    "Unmute realtime returned a non-object event"
                )
            return payload
        if msg.type == aiohttp.WSMsgType.BINARY:
            raise ProviderRunError(
                "Unmute realtime returned an unexpected binary frame"
            )
        if msg.type == aiohttp.WSMsgType.ERROR:
            raise ProviderRunError(
                f"Unmute realtime websocket error: {self._ws.exception()}"
            )
        raise EOFError

    async def close(self) -> None:
        try:
            await self._ws.close()
        finally:
            await self._session.close()


async def _create_aiohttp_websocket(
    url: str,
    headers: dict[str, str],
    timeout: float,
) -> UnmuteProviderSocket:
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, connect=timeout, sock_connect=timeout)
    )
    try:
        # No heartbeat: aiohttp's heartbeat computes `_pong_heartbeat = heartbeat / 2`
        # (so heartbeat=20 → 10s pong deadline) and raises
        # "No PONG received after 10.0 seconds" if the orchestrator (Starlette
        # WebSocket, no auto-pong handler) doesn't reply. The relay isn't an
        # idle long-lived socket — it only stays open while audio is actively
        # flowing — so disabling the heartbeat is safe and avoids the gap.
        ws = await session.ws_connect(url, headers=headers, heartbeat=None)
    except Exception:
        await session.close()
        raise
    return _AioHttpUnmuteProviderSocket(session, ws)


def _session_update(config: RealtimeAgentSessionConfig) -> dict[str, Any]:
    # The unmute backend accepts the OpenAI Realtime session.update shape BUT
    # extends it with two Unmute-specific fields:
    #   * ``instructions`` must be an object (Instructions union: smalltalk |
    #     constant | quiz_show | guess_animal | news | unmute_explanation)
    #     not a free-form string. We default to ``smalltalk`` for English
    #     conversation and let the operator pass ``{type: "constant",
    #     text: "..."}`` via provider_options["instructions"] when needed.
    #   * ``allow_recording`` is required (boolean). Operators opt out by
    #     passing provider_options["allow_recording"]=False.
    # The orchestrator ignores some OpenAI-only fields (output_modalities,
    # model snapshot pinning); we send a minimal valid session.
    transcription_model = _option(config.provider_options, "transcription_model")
    explicit_instructions = config.provider_options.get("instructions")
    if isinstance(explicit_instructions, dict):
        instructions_payload: dict[str, Any] = explicit_instructions
    elif isinstance(explicit_instructions, str) and explicit_instructions.strip():
        instructions_payload = {
            "type": "constant",
            "text": explicit_instructions,
            "language": "en",
        }
    else:
        instructions_payload = {"type": "smalltalk", "language": "en"}
    allow_recording_raw = config.provider_options.get("allow_recording", False)
    allow_recording = bool(allow_recording_raw) if isinstance(allow_recording_raw, (bool, int)) else False
    input_audio: dict[str, Any] = {
        "format": {
            "type": "audio/pcm",
            "rate": config.sample_rate,
        },
        "turn_detection": None,
    }
    if transcription_model:
        input_audio["transcription"] = {"model": transcription_model}
    voice = config.voice or "Watercooler"
    return {
        "type": "session.update",
        "session": {
            "voice": voice,
            "instructions": instructions_payload,
            "allow_recording": allow_recording,
            "audio": {
                "input": input_audio,
                "output": {
                    "format": {"type": "audio/pcm"},
                    "voice": voice,
                },
            },
            "tools": [_openai_function_tool(tool) for tool in config.tools],
            "tool_choice": "auto",
        },
    }


def _openai_function_tool(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": str(schema.get("name") or ""),
        "description": str(schema.get("description") or ""),
        "parameters": dict(schema.get("parameters") or {}),
    }


def _provider_event(
    event: dict[str, Any],
    completed_function_call_ids: set[str],
) -> ProviderEvent | None:
    event_type = str(event.get("type") or "").strip()
    response_id = _response_id(event)
    if event_type in {"session.created", "session.updated", "conversation.created"}:
        session_info = event.get("session")
        resolved_model = (
            str(session_info.get("model") or "").strip()
            if isinstance(session_info, dict)
            else ""
        )
        return ProviderEvent(
            ProviderEventKind.READY,
            response_id=response_id,
            payload={
                "provider_event_type": event_type,
                "resolved_model": resolved_model or None,
            },
        )
    if event_type in {"response.created", "response.output_item.added"}:
        return ProviderEvent(
            ProviderEventKind.RESPONSE_STARTED,
            response_id=response_id,
            payload={"provider_event_type": event_type},
        )
    if event_type in _INPUT_TRANSCRIPT_DELTA_EVENTS:
        delta = _text(event.get("delta") or event.get("transcript") or event.get("text"))
        if delta:
            return ProviderEvent(
                ProviderEventKind.INPUT_TRANSCRIPT_DELTA,
                response_id=response_id,
                payload={"delta": delta},
            )
        return None
    if event_type in _INPUT_TRANSCRIPT_FINAL_EVENTS:
        text = _text(event.get("transcript") or event.get("text") or event.get("delta"))
        if text:
            return ProviderEvent(
                ProviderEventKind.INPUT_TRANSCRIPT_FINAL,
                response_id=response_id,
                payload={"text": text},
            )
        return None
    if event_type in _OUTPUT_TEXT_DELTA_EVENTS:
        delta = _text(event.get("delta") or event.get("text"))
        if delta:
            return ProviderEvent(
                ProviderEventKind.OUTPUT_TEXT_DELTA,
                response_id=response_id,
                payload={"delta": delta},
            )
        return None
    if event_type in _AUDIO_DELTA_EVENTS:
        chunk, audio64 = _decode_audio_delta(event)
        return ProviderEvent(
            ProviderEventKind.AUDIO_DELTA,
            response_id=response_id,
            payload={
                "audio": chunk,
                "audio_base64": audio64,
                "byte_count": len(chunk),
            },
            metadata={"provider_event_type": event_type},
        )
    if event_type in _AUDIO_DONE_EVENTS:
        return ProviderEvent(
            ProviderEventKind.AUDIO_DONE,
            payload={"provider_event_type": event_type},
        )
    if event_type == "response.function_call_arguments.delta":
        return ProviderEvent(
            ProviderEventKind.FUNCTION_CALL_ARGUMENTS_DELTA,
            response_id=response_id,
            payload={
                "call_id": _call_id(event),
                "name": _tool_name(event),
                "delta": _text(event.get("delta") or event.get("arguments")),
            },
        )
    if event_type == "response.function_call_arguments.done":
        event = _event_with_response_output_fallback(event)
        call_id = _call_id(event)
        if call_id:
            completed_function_call_ids.add(call_id)
        return _function_call_event(event, response_id=response_id)
    if event_type in _FUNCTION_ITEM_DONE_EVENTS:
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = _call_id(item)
            if call_id and call_id not in completed_function_call_ids:
                completed_function_call_ids.add(call_id)
                return _function_call_event(item, response_id=response_id)
    if event_type == "response.done":
        function_call = _first_unseen_function_call(event, completed_function_call_ids)
        if function_call is not None:
            call_id = _call_id(function_call)
            if call_id:
                completed_function_call_ids.add(call_id)
            return _function_call_event(function_call, response_id=response_id)
        return ProviderEvent(
            ProviderEventKind.RESPONSE_DONE,
            payload={"provider_event_type": event_type},
        )
    if event_type == "error":
        return ProviderEvent(
            ProviderEventKind.ERROR,
            payload={"message": _format_unmute_error(event)},
        )
    return None


def _function_call_event(event: dict[str, Any], *, response_id: str) -> ProviderEvent:
    name = _tool_name(event)
    arguments = _arguments(event)
    call_id = _call_id(event)
    return ProviderEvent(
        ProviderEventKind.FUNCTION_CALL_COMPLETED,
        response_id=response_id,
        payload={
            "call": ToolCallEvent(
                call_id=call_id,
                name=name,
                arguments=arguments,
            ),
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        },
    )


def _event_with_response_output_fallback(event: dict[str, Any]) -> dict[str, Any]:
    if _tool_name(event) or event.get("arguments"):
        return event
    function_call = _first_function_call(event)
    if function_call is None:
        return event
    merged = dict(function_call)
    for key in ("response_id", "event_id"):
        if key in event:
            merged.setdefault(key, event[key])
    return merged


def _first_unseen_function_call(
    event: dict[str, Any], completed_function_call_ids: set[str]
) -> dict[str, Any] | None:
    for function_call in _function_calls(event):
        call_id = _call_id(function_call)
        if not call_id or call_id not in completed_function_call_ids:
            return function_call
    return None


def _first_function_call(event: dict[str, Any]) -> dict[str, Any] | None:
    return next(iter(_function_calls(event)), None)


def _function_calls(event: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    response = event.get("response")
    if isinstance(response, dict):
        output = response.get("output")
        if isinstance(output, list):
            calls.extend(
                item
                for item in output
                if isinstance(item, dict) and item.get("type") == "function_call"
            )
    item = event.get("item")
    if isinstance(item, dict) and item.get("type") == "function_call":
        calls.append(item)
    if event.get("type") == "function_call":
        calls.append(event)
    return calls


def _decode_audio_delta(event: dict[str, Any]) -> tuple[bytes, str]:
    audio64 = _text(event.get("delta") or event.get("audio"))
    if not audio64:
        raise ProviderRunError("Unmute realtime audio delta missing base64 payload")
    try:
        return base64.b64decode(audio64), audio64
    except Exception as exc:
        raise ProviderRunError(
            "Unmute realtime audio delta was not valid base64"
        ) from exc


def _arguments(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("arguments")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_raw": value}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {}


def _call_id(event: dict[str, Any]) -> str:
    return str(
        event.get("call_id")
        or event.get("item_id")
        or event.get("id")
        or event.get("event_id")
        or ""
    ).strip()


def _tool_name(event: dict[str, Any]) -> str:
    return str(
        event.get("name")
        or event.get("function_name")
        or event.get("tool_name")
        or ""
    ).strip()


def _response_id(event: dict[str, Any]) -> str:
    response = event.get("response")
    if isinstance(response, dict) and response.get("id"):
        return str(response["id"])
    item = event.get("item")
    if isinstance(item, dict) and item.get("response_id"):
        return str(item["response_id"])
    return str(event.get("response_id") or "").strip()


def _format_unmute_error(event: dict[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error.get("type")
        if message:
            return f"Unmute realtime error: {message}"
    return f"Unmute realtime error event: {event}"


def _default_instructions(config: RealtimeAgentSessionConfig) -> str:
    profile = config.profile or "active Hermes profile"
    return (
        "You are the voice front-end for Hermes Relay running through the "
        "Kyutai Unmute realtime backend (Moshi MLX STT/TTS on Mac, Hermes LLM). "
        "Use the provided Hermes functions for user requests that need memory, "
        "tools, device or desktop actions, confirmations, persistent context, "
        "research, current facts, news, external data, live checks, "
        "latest/versioned info, personal or project context, side effects, "
        "precision-sensitive answers, or media and artifact handling. Hermes "
        "is the durable conversation memory; use any seeded recent chat context "
        "for follow-up references when it is enough, and route missing/stale/"
        "verification-sensitive references through Hermes before answering. Do "
        "not say you lack context before a Hermes call. You may speak one brief "
        "acknowledgement such as 'I'll check Hermes' or 'I'll check that' "
        "before the tool call, then call Hermes immediately. Do not give a "
        "substantive answer until Hermes returns. Android will provide "
        "restrained local status while Hermes runs. When speaking, summarize "
        "dense machine-readable values instead of reading raw IDs, URLs, paths, "
        "JSON, logs, or long numbers character by character. "
        f"Active profile: {profile}. "
        "Do not call web_search, x_search, MCP, or any non-Hermes tool."
    )


def _resolve_auth_token(options: dict[str, Any]) -> UnmuteAuthToken:
    """Pick up the kyutai-api-key + optional bearer from provider_options or env.

    Precedence (matches the OpenAI provider's three-tier resolution):

    1. ``provider_options["api_key"]`` (and ``["bearer"]``) — explicit override
    2. ``UNMUTE_API_KEY`` / ``UNMUTE_REALTIME_API_KEY`` env vars
    3. Public default token (the orchestrator's own default auth scheme)
    """
    explicit_key = _option(options, "api_key")
    explicit_bearer = _option(options, "bearer")
    source_key = "default"
    source_bearer = "default"
    api_key = DEFAULT_PUBLIC_TOKEN
    bearer: str | None = None
    if explicit_key:
        api_key = explicit_key
        source_key = "provider-option:api_key"
    else:
        env_key = os.getenv("UNMUTE_REALTIME_API_KEY") or os.getenv("UNMUTE_API_KEY")
        if env_key:
            api_key = env_key.strip()
            source_key = "env:UNMUTE_REALTIME_API_KEY"
    if explicit_bearer:
        bearer = explicit_bearer
        source_bearer = "provider-option:bearer"
    else:
        env_bearer = os.getenv("UNMUTE_REALTIME_BEARER") or os.getenv(
            "UNMUTE_BACKEND_BEARER"
        )
        if env_bearer:
            bearer = env_bearer.strip()
            source_bearer = "env:UNMUTE_REALTIME_BEARER"
    return UnmuteAuthToken(
        bearer=bearer,
        api_key=api_key,
        source=f"api_key={source_key},bearer={source_bearer}",
    )


def _option(options: dict[str, Any], name: str) -> str | None:
    value = options.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_option(options: dict[str, Any], name: str, default: float) -> float:
    value = _option(options, name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ProviderUnavailable(f"{name} must be a number") from exc


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _resample_pcm16_mono(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if not pcm or source_rate <= 0 or target_rate <= 0 or source_rate == target_rate:
        return pcm
    sample_count = len(pcm) // 2
    if sample_count <= 1:
        return pcm
    target_count = max(1, round(sample_count * target_rate / source_rate))
    out = bytearray(target_count * 2)
    for i in range(target_count):
        source_index = min(sample_count - 1, round(i * source_rate / target_rate))
        start = source_index * 2
        out[i * 2 : i * 2 + 2] = pcm[start : start + 2]
    return bytes(out)

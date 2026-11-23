"""Lemonade Realtime provider for Hermes Realtime Agent.

Connects to the local OpenAI-Realtime-compatible bridge running on the
operator's host (default ``ws://127.0.0.1:8010/v1/realtime``). The bridge
forwards:

  * User audio -> Lemonade Whisper-Large-v3-Turbo (server-VAD streaming)
  * Transcript + history -> Lemonade LLM (chat-completions)
  * Reply text -> Lemonade Kokoro TTS
  * TTS audio -> client as ``response.output_audio.delta`` frames

The provider is a thin OpenAI Realtime frame translator. It emits standard
``session.update`` / ``input_audio_buffer.*`` / ``conversation.item.create``
/ ``response.*`` frames; the bridge maps them onto Lemonade backends.

Configuration: ``relay.lemonade.realtime_url`` in ``~/.hermes/config.yaml``
(``LEMONADE_REALTIME_URL`` env var override). No auth required — the bridge
runs on the same host as the relay and accepts a stub ``Authorization``
header.
"""

from __future__ import annotations

import base64
import json
import logging
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
from ..text_sanitize import sanitize_for_tts

_LOG = logging.getLogger("hermes_relay.lemonade_provider")

DEFAULT_MODEL = "Bonsai-1.7B-gguf"
DEFAULT_TRANSCRIPTION_MODEL = "Whisper-Large-v3-Turbo"
DEFAULT_URL = "ws://127.0.0.1:8010/v1/realtime"
DEFAULT_TIMEOUT_SECONDS = 30.0
SUPPORTED_SAMPLE_RATE = 24000

# Frame sets — same names as the OpenAI provider because the bridge speaks
# OpenAI Realtime; if the bridge renames an event, this is the only place
# to update.
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


class LemonadeProviderSocket(Protocol):
    async def send_json(self, payload: dict[str, Any]) -> None: ...

    async def receive_json(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


SocketFactory = Callable[[str, dict[str, str], float], Any]


class LemonadeRealtimeAgentAdapter(RealtimeAgentProviderAdapter):
    """Render-only lemonade adapter kept for the non-native fallback path."""

    provider_id = "lemonade_realtime"


class LemonadeRealtimeAgentProvider:
    """Connection-oriented Lemonade Realtime provider.

    Speaks the OpenAI Realtime WebSocket protocol to a local Lemonade bridge
    that drives STT → chat-completions (with native function-call support
    for Hermes tools) → TTS. The broker's tool loop is enabled: when the
    bridge emits ``response.function_call_arguments.done`` events, the
    broker intercepts them, runs the matching Hermes tool, and sends the
    result back via ``conversation.item.create(function_call_output)``,
    which the bridge folds back into the chat-completions history.
    """

    provider_id = "lemonade_realtime"
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
    ) -> "LemonadeRealtimeAgentConnection":
        if config.sample_rate != SUPPORTED_SAMPLE_RATE:
            raise ProviderUnavailable(
                "Lemonade realtime currently supports sample_rate=24000 for "
                "provider-native Realtime Agent sessions."
            )
        timeout = _float_option(
            config.provider_options, "timeout", DEFAULT_TIMEOUT_SECONDS
        )
        base_url = (
            _option(config.provider_options, "url")
            or os.getenv("LEMONADE_REALTIME_URL")
            or DEFAULT_URL
        ).rstrip("/")
        # Pass model + voice as query params so the bridge can read them
        # without depending on the operator's session.update frame.
        explicit_model = _option(config.provider_options, "model") or (
            config.model or DEFAULT_MODEL
        )
        separator = "&" if "?" in base_url else "?"
        url = f"{base_url}{separator}model={urllib.parse.quote(explicit_model, safe='')}"

        # Bridge accepts (and ignores) any Authorization header.
        headers = {"Authorization": "Bearer local"}

        try:
            socket = await self._socket_factory(url, headers, timeout)
        except aiohttp.WSServerHandshakeError as exc:
            raise ProviderUnavailable(
                f"Lemonade realtime bridge at {url} rejected the handshake "
                f"(HTTP {exc.status}). Confirm the bridge is running "
                "(realtime-bridge/scripts/run.sh)."
            ) from exc
        connection = LemonadeRealtimeAgentConnection(
            socket=socket,
            config=config,
        )
        await connection.configure()
        return connection


class LemonadeRealtimeAgentConnection:
    def __init__(
        self,
        *,
        socket: LemonadeProviderSocket,
        config: RealtimeAgentSessionConfig,
    ) -> None:
        self.socket = socket
        self.config = config
        self._completed_function_call_ids: set[str] = set()
        # Counters for frame-in / frame-out so debug logs reveal stalled
        # providers (e.g. bridge dead, response.done never arrives) at a
        # glance. Reset on each ``configure()``.
        self._tx_frame_count: int = 0
        self._rx_frame_count: int = 0

    def _log_frame(self, direction: str, payload: dict[str, Any]) -> None:
        """Single entry-point for all WS traffic so debug logs are uniform."""
        # Count audio deltas by byte, not by event, so the log doesn't
        # explode during normal speech playback.
        if payload.get("type") == "response.output_audio.delta" and direction == "rx":
            self._rx_frame_count += 1
            if self._rx_frame_count % 10 == 1:
                _LOG.debug(
                    "rx[%d] %s (audio_delta; total deltas=%d)",
                    self._rx_frame_count,
                    direction,
                    self._rx_frame_count,
                )
            return
        if direction == "rx":
            self._rx_frame_count += 1
            n = self._rx_frame_count
        else:
            self._tx_frame_count += 1
            n = self._tx_frame_count
        # Strip base64 audio from log lines — would otherwise flood stdout.
        safe = {
            k: v
            for k, v in payload.items()
            if k not in {"audio", "audio_base64"}
        }
        _LOG.debug("frame %s tx=%d rx=%d: %s", direction, self._tx_frame_count, self._rx_frame_count, safe)

    async def configure(self) -> None:
        session_update = _session_update(self.config)
        self._log_frame("tx", session_update)
        await self.socket.send_json(session_update)
        # Trigger the bridge's first response (the opening greeting). The
        # Lemonade bridge only fires a response when it receives
        # response.create or input_audio_buffer.commit — sending
        # session.update alone leaves the user staring at silence on
        # connection. The OpenAI and xAI providers do the same.
        #
        # Operators can suppress this via
        # ``relay.lemonade.greeting_enabled: false`` in Hermes config
        # (forwarded into provider_options["greeting_enabled"]) when the
        # client side handles its own post-connect state transitions
        # and the spoken greeting causes more harm than benefit (e.g.
        # post-greeting Speaking-state races in the Android app).
        if _bool_option(self.config.provider_options, "greeting_enabled", True):
            create_frame = {"type": "response.create"}
            self._log_frame("tx", create_frame)
            await self.socket.send_json(create_frame)
        _LOG.info(
            "lemonade_provider configured: model=%s voice=%s tools=%d sample_rate=%d greeting_enabled=%s",
            self.config.model,
            self.config.voice,
            len(self.config.tools),
            self.config.sample_rate,
            _bool_option(self.config.provider_options, "greeting_enabled", True),
        )

    async def send_audio(self, pcm: bytes, sample_rate: int) -> None:
        pcm = _resample_pcm16_mono(pcm, sample_rate, self.config.sample_rate)
        frame = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }
        # Audio appends are very chatty — log a sampled line every 20 frames
        # rather than every one. The actual byte count is what matters.
        self._tx_frame_count += 1
        if self._tx_frame_count % 20 == 1:
            _LOG.debug(
                "tx audio append: tx=%d bytes=%d rate=%d",
                self._tx_frame_count, len(pcm), self.config.sample_rate,
            )
        await self.socket.send_json(frame)

    async def commit_audio(self) -> None:
        frame = {"type": "input_audio_buffer.commit"}
        self._log_frame("tx", frame)
        await self.socket.send_json(frame)
        _LOG.info("tx input_audio_buffer.commit — request completion of buffered audio")
        # NOTE: do NOT emit response.create here. The bridge auto-triggers a
        # response from its STT finalization event
        # (conversation.item.input_audio_transcription.completed →
        # _run_response). Sending response.create additionally would race the
        # single-flight guard and double-trigger the LLM, producing
        # "offset-by-one" responses that repeat the previous turn.

    async def send_text(self, text: str) -> None:
        frame = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
        self._log_frame("tx", frame)
        await self.socket.send_json(frame)
        create = {"type": "response.create"}
        self._log_frame("tx", create)
        await self.socket.send_json(create)

    async def clear_audio(self) -> None:
        frame = {"type": "input_audio_buffer.clear"}
        self._log_frame("tx", frame)
        await self.socket.send_json(frame)

    async def cancel_response(self) -> None:
        frame = {"type": "response.cancel"}
        self._log_frame("tx", frame)
        await self.socket.send_json(frame)
        _LOG.info("tx response.cancel — bridge mid-response aborted")

    async def send_tool_result(self, call_id: str, output: dict[str, Any]) -> None:
        # Function-call surface IS available — the bridge passes tools to
        # Lemonade chat-completions and emits response.function_call_arguments.done
        # events which the broker intercepts to run Hermes tools. Forward
        # the broker's tool output back to the bridge as an OpenAI Realtime
        # function_call_output item so the next chat-completions request
        # can resume the agentic loop.
        # Hermes tool results come back as {answer, text, summary, ...};
        # the bridge wants a string (OpenAI Realtime spec) so we flatten.
        answer_text = (
            str(output.get("answer") or output.get("text") or output.get("summary") or "").strip()
        )
        item_frame = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": answer_text or json.dumps(output, ensure_ascii=False),
            },
        }
        self._log_frame("tx", item_frame)
        await self.socket.send_json(item_frame)
        # And trigger a fresh response so the bridge resumes.
        create = {"type": "response.create"}
        self._log_frame("tx", create)
        await self.socket.send_json(create)
        _LOG.info("tx function_call_output for call_id=%s -> response.create", call_id)

    async def append_context_item(self, *, role: str, text: str) -> None:
        content_type = "text" if role == "assistant" else "input_text"
        await self.socket.send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": role,
                    "content": [{"type": content_type, "text": text}],
                },
            }
        )

    async def request_response(
        self,
        *,
        instructions: str | None = None,
        exact_text: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"type": "response.create"}
        if instructions:
            payload["response"] = {"instructions": instructions}
        self._log_frame("tx", payload)
        await self.socket.send_json(payload)

    async def close(self) -> None:
        _LOG.info("lemonade_provider closing: tx=%d rx=%d", self._tx_frame_count, self._rx_frame_count)
        await self.socket.close()

    async def events(self) -> AsyncIterator[ProviderEvent]:
        while True:
            try:
                raw = await self.socket.receive_json()
            except (EOFError, StopAsyncIteration):
                _LOG.info("lemonade_provider WS closed: tx=%d rx=%d", self._tx_frame_count, self._rx_frame_count)
                return
            except ProviderRunError as exc:
                _LOG.warning("lemonade_provider WS error: %s", exc)
                yield ProviderEvent(
                    ProviderEventKind.ERROR,
                    payload={"message": str(exc)},
                )
                return
            # Log every non-audio frame and a sampled subset of audio deltas.
            self._log_frame("rx", raw if isinstance(raw, dict) else {"type": "<non-dict>"})
            event = _provider_event(raw, self._completed_function_call_ids)
            if event is not None:
                yield event


class _AioHttpLemonadeProviderSocket:
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
                    "Lemonade realtime bridge returned non-JSON text"
                ) from exc
            if not isinstance(payload, dict):
                raise ProviderRunError(
                    "Lemonade realtime bridge returned a non-object event"
                )
            return payload
        if msg.type == aiohttp.WSMsgType.BINARY:
            raise ProviderRunError(
                "Lemonade realtime bridge returned an unexpected binary frame"
            )
        if msg.type == aiohttp.WSMsgType.ERROR:
            raise ProviderRunError(
                f"Lemonade realtime bridge websocket error: {self._ws.exception()}"
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
) -> LemonadeProviderSocket:
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, connect=timeout, sock_connect=timeout)
    )
    try:
        ws = await session.ws_connect(
            url,
            headers=headers,
            # Realtime audio keeps the WS warm; disable aiohttp's
            # bidirectional heartbeat to avoid 1011 keepalive ping timeouts
            # mid-stream.
            heartbeat=None,
        )
    except aiohttp.WSServerHandshakeError:
        await session.close()
        raise
    except Exception:
        await session.close()
        raise
    return _AioHttpLemonadeProviderSocket(session, ws)


def _is_broker_bureaucracy(instructions: str) -> bool:
    """Detect the broker's full speech-router prompt so we can drop it.

    The broker's ``_native_instructions`` is a 5000+ char system prompt meant
    for hosted GPT-4o / grok. Local models (Bonsai 1.7B, Qwen3.5-4B) hallucinate
    'Hermes Relay architecture' descriptions when fed it. Match the leader
    string — anything starting with this is the full prompt, not a persona.
    """
    if not instructions:
        return False
    return instructions.startswith(
        "You are the provider-native speech loop for Hermes Relay"
    )


def _build_instructions(config: RealtimeAgentSessionConfig) -> str:
    """Pass Hermes's authoritative system prompt through verbatim.

    The broker's ``_native_instructions(session)` already encodes the full
    Hermes Relay voice contract: which tools to call (hermes_run_task,
    hermes_get_status, hermes_cancel, hermes_confirm), how to answer date /
    time questions from relay-local context, when to skip a re-run, and how
    Hermes owns memory / confirmations / persistence. The provider must not
    rewrite, summarize, or fragment it.

    Local-model handling note: Bonsai 1.7B / Qwen3.5-4B used to choke on the
    full prompt and hallucinate architecture descriptions. The current
    deployment runs Qwen3.6-35B-A3B (or larger), which can absorb the
    contract. If a future deployment drops back to a tiny local model, this
    pass-through is the place to re-introduce the size guard — gated on
    ``config.model`` rather than rewriting Hermes's prompt.
    """
    raw = (config.instructions or "").strip()
    explicit = _option(config.provider_options, "instructions")

    if explicit:
        # Operator override via provider_options["instructions"] — used for
        # one-off Voice Lab experiments. Takes precedence over the broker's
        # prompt.
        return explicit
    if raw:
        return raw
    # No prompt arrived from Hermes and no override was set. Don't fabricate
    # one — surface the gap so the operator notices.
    return (
        "You are a real-time voice assistant. No system prompt was supplied "
        "by the relay; speak naturally, keep replies short, and call "
        "hermes_run_task for any question that needs current state or tools."
    )


def _session_update(config: RealtimeAgentSessionConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "emotion": _option(config.provider_options, "emotion"),
        "tone": _option(config.provider_options, "tone"),
        "intensity": _option(config.provider_options, "intensity"),
        "pace": _option(config.provider_options, "pace"),
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return {
        "type": "session.update",
        "session": {
            "model": config.model or DEFAULT_MODEL,
            "voice": config.voice,
            "instructions": _build_instructions(config),
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": config.sample_rate},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": config.sample_rate},
                    "voice": config.voice,
                },
            },
            # Hermes tool surface — chat-completions tool-call format
            # ({type: "function", name, description, parameters}). The
            # bridge passes these straight to Lemonade chat-completions
            # and surfaces tool_calls as OpenAI Realtime function-call
            # events for the broker to consume.
            "tools": [_lemonade_function_tool(tool) for tool in config.tools],
            "tool_choice": "auto",
            "metadata": metadata,
        },
    }


def _lemonade_function_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a Hermes tool schema in the chat-completions function shape."""
    tool = dict(schema)
    tool.setdefault("type", "function")
    return tool


# Lemonade bridge frame field extraction helpers. The bridge emits
# OpenAI-Realtime-shaped frames but the function-call identifying fields
# vary across frames (top-level vs nested under ``item`` vs ``response``);
# the canonical unmute.py implementation already has equivalents — these
# are scoped to the lemonade bridge's actual emit shape.
def _lemonade_call_id(event: dict[str, Any]) -> str:
    """Pull call_id out of the frame regardless of where the bridge nested it.

    The bridge emits ``call_id`` at the top level for
    ``response.function_call_arguments.done`` and nested under ``item``
    for ``response.output_item.added/.done``.

    The bridge prefixes WS item ids with ``fc_`` (see ``bridge/server.py``
    ``f"fc_{tc.id}"``), but the inner ``tc.id`` — the one the bridge
    actually keys ``pending_tool_results`` against — has no prefix. Forward
    the un-prefixed id back as ``function_call_output.call_id`` so the
    bridge's ``_lookup_tool_name`` and chat-completions tool-result
    pairing match. Without the strip the relay sends ``call_id="fc_xxx"``
    and the bridge falls back to name="tool" because no history entry
    matches the prefixed id.
    """
    item = event.get("item")
    if isinstance(item, dict):
        for key in ("call_id", "id", "item_id"):
            value = item.get(key)
            if value:
                return str(value).strip().removeprefix("fc_")
    for key in ("call_id", "item_id", "id"):
        value = event.get(key)
        if value:
            return str(value).strip().removeprefix("fc_")
    return ""


def _lemonade_tool_name(event: dict[str, Any]) -> str:
    item = event.get("item")
    if isinstance(item, dict):
        for key in ("name", "function_name", "tool_name"):
            value = item.get(key)
            if value:
                return str(value).strip()
    for key in ("name", "function_name", "tool_name"):
        value = event.get(key)
        if value:
            return str(value).strip()
    return ""


def _lemonade_arguments(event: dict[str, Any]) -> dict[str, Any]:
    """Parse arguments either as a JSON string or already-decoded dict."""
    item = event.get("item")
    raw: Any = None
    if isinstance(item, dict) and item.get("type") == "function_call":
        raw = item.get("arguments")
    if raw is None:
        raw = event.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {}


def _lemonade_function_calls(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract function_call items from ``response.done.response.output``
    or from ``event.item`` for direct-shaped frames.
    """
    calls: list[dict[str, Any]] = []
    response = event.get("response")
    if isinstance(response, dict):
        output = response.get("output")
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and item.get("type") == "function_call":
                    calls.append(item)
    item = event.get("item")
    if isinstance(item, dict) and item.get("type") == "function_call":
        calls.append(item)
    if event.get("type") == "function_call":
        calls.append(event)
    return calls


def _first_unseen_function_call(
    event: dict[str, Any],
    completed_function_call_ids: set[str],
) -> dict[str, Any] | None:
    for function_call in _lemonade_function_calls(event):
        call_id = _lemonade_call_id(function_call)
        if not call_id or call_id not in completed_function_call_ids:
            return function_call
    return None


def _function_call_added_event(event: dict[str, Any], response_id: str) -> ProviderEvent:
    """``response.output_item.added`` for a function_call item.

    The broker uses RESPONSE_STARTED to start a UI playback cycle; on a
    function-call item we don't want to do that (there is no audio yet).
    Surface as FUNCTION_CALL_STARTED so the broker can record the call_id
    in flight without firing the playback machinery.
    """
    return ProviderEvent(
        ProviderEventKind.FUNCTION_CALL_STARTED,
        response_id=response_id,
        payload={
            "call_id": _lemonade_call_id(event),
            "name": _lemonade_tool_name(event),
            "provider_event_type": "response.output_item.added",
        },
    )


def _function_call_event(event: dict[str, Any], response_id: str) -> ProviderEvent:
    """Construct a FUNCTION_CALL_COMPLETED provider event with the broker's
    expected payload shape (call_id / name / arguments / call).
    """
    call_id = _lemonade_call_id(event)
    name = _lemonade_tool_name(event)
    arguments = _lemonade_arguments(event)
    _LOG.info(
        "rx function_call: call_id=%s name=%s args=%s",
        call_id, name, json.dumps(arguments, ensure_ascii=False)[:200],
    )
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
            "provider_event_type": "response.function_call_arguments.done",
        },
    )


def _provider_event(
    event: dict[str, Any],
    completed_function_call_ids: set[str],
) -> ProviderEvent | None:
    event_type = str(event.get("type") or "").strip()
    response_id = str(event.get("response_id") or "").strip()

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

    if event_type in {"response.created"}:
        return ProviderEvent(
            ProviderEventKind.RESPONSE_STARTED,
            response_id=response_id,
            payload={"provider_event_type": event_type},
        )

    if event_type == "response.output_item.added":
        # The bridge emits this for both assistant text/audio items AND
        # function_call items. Distinguish by item.type so the broker's
        # RESPONSE_STARTED path doesn't fire prematurely on a function
        # call (which has no audio and isn't a real assistant turn yet).
        item = event.get("item") or {}
        if isinstance(item, dict) and item.get("type") == "function_call":
            return _function_call_added_event(event, response_id)
        return ProviderEvent(
            ProviderEventKind.RESPONSE_STARTED,
            response_id=response_id,
            payload={"provider_event_type": event_type},
        )

    if event_type == "response.function_call_arguments.delta":
        return ProviderEvent(
            ProviderEventKind.FUNCTION_CALL_ARGUMENTS_DELTA,
            response_id=response_id,
            payload={
                "call_id": _lemonade_call_id(event),
                "name": _lemonade_tool_name(event),
                "delta": _text(event.get("delta") or event.get("arguments")),
            },
        )

    if event_type == "response.function_call_arguments.done":
        call_id = _lemonade_call_id(event)
        if call_id:
            completed_function_call_ids.add(call_id)
        return _function_call_event(event, response_id)

    if event_type in _FUNCTION_ITEM_DONE_EVENTS:
        # ``response.output_item.done`` for a function_call item is the
        # canonical signal the bridge emits. Cover both shapes — the
        # function-call fields live under ``item`` for ``output_item.done``
        # and under the top level for ``function_call_arguments.done``.
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = _lemonade_call_id(item)
            if call_id and call_id not in completed_function_call_ids:
                completed_function_call_ids.add(call_id)
                return _function_call_event(item, response_id)
        return None

    if event_type == "response.done":
        # Final fallback: if neither of the above fired, the response.done
        # frame may still carry the function_call under ``response.output``.
        # We only synthesize a FUNCTION_CALL_COMPLETED event from this
        # path if we haven't already seen one for the same call_id —
        # otherwise we'd double-fire the broker tool loop.
        function_call = _first_unseen_function_call(event, completed_function_call_ids)
        if function_call is not None:
            call_id = _lemonade_call_id(function_call)
            if call_id:
                completed_function_call_ids.add(call_id)
            return _function_call_event(function_call, response_id)
        return ProviderEvent(
            ProviderEventKind.RESPONSE_DONE,
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
        text = _text(
            event.get("transcript") or event.get("text") or event.get("delta")
        )
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
            clean = sanitize_for_tts(delta)
            if not clean:
                # Pure punctuation / markup — drop the event so the broker
                # doesn't broadcast empty deltas to the phone UI.
                _LOG.debug("lemonade_provider dropped empty output text delta after sanitize")
                return None
            if clean != delta:
                _LOG.debug("lemonade_provider sanitized output delta: in=%r out=%r", delta, clean)
            return ProviderEvent(
                ProviderEventKind.OUTPUT_TEXT_DELTA,
                response_id=response_id,
                payload={"delta": clean},
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
            response_id=response_id,
            payload={"provider_event_type": event_type},
        )

    if event_type == "response.done":
        return ProviderEvent(
            ProviderEventKind.RESPONSE_DONE,
            response_id=response_id,
            payload={"provider_event_type": event_type},
        )

    if event_type == "error":
        return ProviderEvent(
            ProviderEventKind.ERROR,
            response_id=response_id,
            payload={"message": _format_bridge_error(event)},
        )

    # Bridge-specific events the broker doesn't model yet — surface as
    # output text so the Android UI can render them. ``voice.input_transcript.final``
    # is the bridge's name for the *post-VAD* user transcript (the
    # openai-style input_audio_transcription events are also emitted; this
    # one carries the post-commit final form).
    if event_type in {"voice.input_transcript.final"}:
        text = _text(event.get("text"))
        if text:
            return ProviderEvent(
                ProviderEventKind.INPUT_TRANSCRIPT_FINAL,
                response_id=response_id,
                payload={"text": text, "source": "voice_input_transcript_final"},
            )
        return None

    return None


def _decode_audio_delta(event: dict[str, Any]) -> tuple[bytes, str]:
    payload = event.get("delta") or event.get("audio")
    if not isinstance(payload, str):
        raise ProviderRunError("Lemonade realtime audio delta missing base64 payload")
    try:
        chunk = base64.b64decode(payload)
    except Exception as exc:
        raise ProviderRunError(
            "Lemonade realtime audio delta was not valid base64"
        ) from exc
    return chunk, payload


def _format_bridge_error(event: dict[str, Any]) -> str:
    err = event.get("error")
    if isinstance(err, dict):
        message = err.get("message") or err.get("code") or err.get("type")
        if message:
            return str(message)
    return "Unknown bridge error"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _resample_pcm16_mono(pcm: bytes, in_rate: int, out_rate: int) -> bytes:
    """Linear resampler for s16le mono PCM. Returns input unchanged on
    rate mismatch (the bridge will re-interpret).

    Lemond exposes a VAD/stream that demands 24 kHz; the unmute provider
    uses a heavier ``sphn`` resampler, but for our 24 kHz-only path the
    rate is invariant for the realtime-agent case.
    """
    if in_rate == out_rate or not pcm:
        return pcm
    # Cheap path: nearest-neighbour resample. Sufficient for 24 kHz → 24 kHz
    # (no-ops) and 16 kHz → 24 kHz (1.5x up).
    import audioop

    return audioop.ratecv(pcm, 2, 1, in_rate, out_rate, None)[0]


def _option(options: dict[str, Any], name: str) -> str | None:
    value = options.get(name)
    if value is None or value == "":
        return None
    return str(value)


def _bool_option(options: dict[str, Any], name: str, default: bool) -> bool:
    """Read a boolean option, accepting strings ("false"/"true"/etc.) and real booleans.

    Mirrors :func:`_option`'s forgiving coercion. Defaults to ``default`` when
    the key is absent so callers can opt-out of behavior with an explicit
    ``False`` while keeping the legacy default intact.
    """
    raw = options.get(name)
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"false", "0", "no", "off"}:
            return False
        if lowered in {"true", "1", "yes", "on"}:
            return True
    return bool(raw)


def _float_option(options: dict[str, Any], name: str, default: float) -> float:
    raw = options.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default

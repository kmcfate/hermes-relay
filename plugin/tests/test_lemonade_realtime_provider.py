"""Tests for the Lemonade provider-native realtime-agent adapter.

The Lemonade provider talks to a local realtime bridge that speaks the
OpenAI Realtime WebSocket protocol. Unlike the Unmute provider, the
bridge accepts the full OpenAI client-frame vocabulary, so the lifecycle
methods are real frames rather than no-ops. The quirks worth locking
down here:

* the bridge only fires a response on ``response.create`` or
  ``input_audio_buffer.commit`` — a bare ``session.update`` leaves the
  user in silence, so ``configure()`` sends a greeting ``response.create``
  unless ``provider_options["greeting_enabled"]`` is false
* ``commit_audio()`` sends ONLY ``input_audio_buffer.commit``; adding a
  ``response.create`` would race the bridge's single-flight guard and
  double-trigger the LLM ("offset-by-one" repeated turns)
* tool results are forwarded as a ``conversation.item.create`` with a
  ``function_call_output`` item, then a ``response.create``
* the bridge prefixes WS item ids with ``fc_``, but the chat-completions
  tool-call id it keys pending results against has no prefix — the relay
  strips the prefix so tool-result pairing resolves
* Hermes's authoritative system prompt is passed through verbatim
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import unittest
from typing import Any
from unittest.mock import patch

from plugin.relay.realtime_agent.models import (
    ProviderEventKind,
    RealtimeAgentSessionConfig,
)
from plugin.relay.realtime_agent.providers.lemonade import (
    DEFAULT_MODEL,
    DEFAULT_URL,
    LemonadeRealtimeAgentProvider,
)
from plugin.voice_lab.providers.base import ProviderUnavailable


class FakeLemonadeSocket:
    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self._events = list(events or [])
        self.closed = False

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def receive_json(self) -> dict[str, Any]:
        if not self._events:
            raise EOFError
        return self._events.pop(0)

    async def close(self) -> None:
        self.closed = True


@contextlib.contextmanager
def clean_lemonade_env():
    """Drop every ``LEMONADE_*`` env var so URL resolution is deterministic."""
    with patch.dict(os.environ):
        for key in [k for k in os.environ if k.startswith("LEMONADE_")]:
            os.environ.pop(key, None)
        yield


def _make_config(**overrides: Any) -> RealtimeAgentSessionConfig:
    base: dict[str, Any] = {
        "provider": "lemonade_realtime",
        "model": "Bonsai-1.7B-gguf",
        "voice": "default",
        "sample_rate": 24000,
        "profile": "default",
        "hermes_session_id": None,
        "instructions": "",
    }
    base.update(overrides)
    return RealtimeAgentSessionConfig(**base)


class LemonadeRealtimeAgentProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_id_and_capabilities(self) -> None:
        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return FakeLemonadeSocket()

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        self.assertEqual(provider.provider_id, "lemonade_realtime")
        caps = provider.capabilities
        self.assertEqual(caps.provider_id, "lemonade_realtime")
        self.assertTrue(caps.supports_function_calls)
        self.assertTrue(caps.supports_server_vad)

    async def test_default_url_carries_model_query_param(self) -> None:
        seen: dict[str, Any] = {}

        async def _factory(
            url: str, headers: dict[str, str], timeout: float
        ) -> FakeLemonadeSocket:
            seen["url"] = url
            seen["headers"] = headers
            return FakeLemonadeSocket(events=[{"type": "response.done"}])

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            await provider.connect(_make_config(model="Qwen3.5-35B-A3B"))

        self.assertTrue(seen["url"].startswith(DEFAULT_URL))
        self.assertIn("model=Qwen3.5-35B-A3B", seen["url"])
        # The bridge accepts and ignores any Authorization header.
        self.assertEqual(seen["headers"]["Authorization"], "Bearer local")

    async def test_default_model_used_when_config_model_empty(self) -> None:
        seen: dict[str, Any] = {}

        async def _factory(
            url: str, headers: dict[str, str], timeout: float
        ) -> FakeLemonadeSocket:
            seen["url"] = url
            return FakeLemonadeSocket(events=[{"type": "response.done"}])

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            await provider.connect(_make_config(model=""))
        self.assertIn(f"model={DEFAULT_MODEL}", seen["url"])

    async def test_env_url_used_when_no_provider_option(self) -> None:
        seen: dict[str, Any] = {}

        async def _factory(
            url: str, headers: dict[str, str], timeout: float
        ) -> FakeLemonadeSocket:
            seen["url"] = url
            return FakeLemonadeSocket(events=[{"type": "response.done"}])

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            os.environ["LEMONADE_REALTIME_URL"] = (
                "ws://lemonade.example.test:8010/v1/realtime"
            )
            await provider.connect(_make_config())
        self.assertTrue(
            seen["url"].startswith("ws://lemonade.example.test:8010/v1/realtime")
        )

    async def test_provider_option_url_wins_over_env(self) -> None:
        seen: dict[str, Any] = {}

        async def _factory(
            url: str, headers: dict[str, str], timeout: float
        ) -> FakeLemonadeSocket:
            seen["url"] = url
            return FakeLemonadeSocket(events=[{"type": "response.done"}])

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        config = _make_config()
        config.provider_options["url"] = "ws://option.example.test:8010/v1/realtime"
        with clean_lemonade_env():
            os.environ["LEMONADE_REALTIME_URL"] = (
                "ws://env.example.test:8010/v1/realtime"
            )
            await provider.connect(config)
        self.assertTrue(
            seen["url"].startswith("ws://option.example.test:8010/v1/realtime")
        )

    async def test_unsupported_sample_rate_rejected(self) -> None:
        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return FakeLemonadeSocket()

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            with self.assertRaisesRegex(ProviderUnavailable, "sample_rate=24000"):
                await provider.connect(_make_config(sample_rate=16000))

    async def test_session_update_shape(self) -> None:
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(
                _make_config(model="Bonsai-1.7B-gguf", voice="default")
            )
        try:
            session_update = socket.sent[0]
            self.assertEqual(session_update["type"], "session.update")
            session = session_update["session"]
            self.assertEqual(session["model"], "Bonsai-1.7B-gguf")
            self.assertEqual(session["voice"], "default")
            self.assertEqual(
                session["audio"]["input"]["format"],
                {"type": "audio/pcm", "rate": 24000},
            )
            self.assertEqual(
                session["audio"]["output"]["format"],
                {"type": "audio/pcm", "rate": 24000},
            )
            self.assertEqual(session["audio"]["output"]["voice"], "default")
            self.assertEqual(session["tool_choice"], "auto")
            self.assertIsInstance(session["tools"], list)
        finally:
            await connection.close()

    async def test_configure_sends_greeting_response_create(self) -> None:
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            types = [msg["type"] for msg in socket.sent]
            self.assertEqual(types[0], "session.update")
            self.assertIn("response.create", types)
        finally:
            await connection.close()

    async def test_greeting_can_be_disabled(self) -> None:
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        config = _make_config()
        config.provider_options["greeting_enabled"] = False
        with clean_lemonade_env():
            connection = await provider.connect(config)
        try:
            types = [msg["type"] for msg in socket.sent]
            self.assertEqual(types, ["session.update"])
        finally:
            await connection.close()

    async def test_commit_audio_sends_only_commit_frame(self) -> None:
        """Adding ``response.create`` here would double-trigger the bridge."""
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            socket.sent.clear()
            await connection.commit_audio()
            self.assertEqual(socket.sent, [{"type": "input_audio_buffer.commit"}])
        finally:
            await connection.close()

    async def test_send_audio_base64_encodes_pcm(self) -> None:
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            socket.sent.clear()
            pcm = b"\x10\x20\x30\x40\x50\x60"
            await connection.send_audio(pcm, sample_rate=24000)
            self.assertEqual(len(socket.sent), 1)
            msg = socket.sent[0]
            self.assertEqual(msg["type"], "input_audio_buffer.append")
            self.assertEqual(msg["audio"], base64.b64encode(pcm).decode("ascii"))
        finally:
            await connection.close()

    async def test_send_tool_result_flattens_output_and_requests_response(self) -> None:
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            socket.sent.clear()
            await connection.send_tool_result("call_1", {"answer": "42"})
            self.assertEqual(len(socket.sent), 2)
            item_frame = socket.sent[0]
            self.assertEqual(item_frame["type"], "conversation.item.create")
            self.assertEqual(item_frame["item"]["type"], "function_call_output")
            self.assertEqual(item_frame["item"]["call_id"], "call_1")
            self.assertEqual(item_frame["item"]["output"], "42")
            self.assertEqual(socket.sent[1], {"type": "response.create"})
        finally:
            await connection.close()

    async def test_send_tool_result_falls_back_to_json_dump(self) -> None:
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            socket.sent.clear()
            await connection.send_tool_result("call_1", {"detail": "raw"})
            self.assertEqual(
                socket.sent[0]["item"]["output"], json.dumps({"detail": "raw"})
            )
        finally:
            await connection.close()

    async def test_instructions_pass_hermes_prompt_verbatim(self) -> None:
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        prompt = "You are the voice front-end for Hermes Relay. Call hermes_run_task."
        with clean_lemonade_env():
            connection = await provider.connect(_make_config(instructions=prompt))
        try:
            self.assertEqual(socket.sent[0]["session"]["instructions"], prompt)
        finally:
            await connection.close()

    async def test_provider_option_instructions_override_hermes_prompt(self) -> None:
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        config = _make_config(instructions="hermes prompt")
        config.provider_options["instructions"] = "operator override"
        with clean_lemonade_env():
            connection = await provider.connect(config)
        try:
            self.assertEqual(
                socket.sent[0]["session"]["instructions"], "operator override"
            )
        finally:
            await connection.close()

    async def test_instructions_fallback_when_no_prompt_supplied(self) -> None:
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config(instructions=""))
        try:
            instructions = socket.sent[0]["session"]["instructions"]
            self.assertIsInstance(instructions, str)
            self.assertIn("real-time voice assistant", instructions)
        finally:
            await connection.close()

    async def test_call_id_prefix_is_stripped(self) -> None:
        """The bridge keys pending tool results by the un-prefixed id."""
        socket = FakeLemonadeSocket(
            events=[
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "id": "fc_abc123",
                        "name": "hermes_run_task",
                        "arguments": "{}",
                    },
                },
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "abc123",
                    "name": "hermes_run_task",
                    "arguments": json.dumps({"text": "hi"}),
                },
                {"type": "response.done"},
            ]
        )

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            events: list[Any] = []
            async for event in connection.events():
                events.append(event)
            started = [
                e for e in events if e.kind == ProviderEventKind.FUNCTION_CALL_STARTED
            ]
            completed = [
                e for e in events if e.kind == ProviderEventKind.FUNCTION_CALL_COMPLETED
            ]
            self.assertEqual(len(started), 1)
            self.assertEqual(started[0].payload["call_id"], "abc123")
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0].payload["call_id"], "abc123")
            self.assertEqual(completed[0].payload["name"], "hermes_run_task")
            self.assertEqual(completed[0].payload["arguments"], {"text": "hi"})
        finally:
            await connection.close()

    async def test_function_call_emitted_once_per_call_id(self) -> None:
        socket = FakeLemonadeSocket(
            events=[
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "call_1",
                    "name": "hermes_run_task",
                    "arguments": json.dumps({"text": "hi"}),
                },
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "id": "call_1",
                        "name": "hermes_run_task",
                        "arguments": json.dumps({"text": "hi"}),
                    },
                },
                {"type": "response.done"},
            ]
        )

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            events: list[Any] = []
            async for event in connection.events():
                events.append(event)
            completed = [
                e for e in events if e.kind == ProviderEventKind.FUNCTION_CALL_COMPLETED
            ]
            self.assertEqual(len(completed), 1)
        finally:
            await connection.close()

    async def test_output_text_delta_sanitized_for_tts(self) -> None:
        socket = FakeLemonadeSocket(
            events=[
                {"type": "response.output_text.delta", "delta": "**bold** answer"},
                {"type": "response.done"},
            ]
        )

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            events: list[Any] = []
            async for event in connection.events():
                events.append(event)
            deltas = [
                e for e in events if e.kind == ProviderEventKind.OUTPUT_TEXT_DELTA
            ]
            self.assertEqual(len(deltas), 1)
            self.assertNotIn("*", deltas[0].payload["delta"])
            self.assertIn("bold", deltas[0].payload["delta"])
        finally:
            await connection.close()

    async def test_audio_delta_event_decoded_to_pcm(self) -> None:
        pcm = b"\xff\xfe\xfd\xfc"
        encoded = base64.b64encode(pcm).decode("ascii")
        socket = FakeLemonadeSocket(
            events=[
                {"type": "response.output_audio.delta", "delta": encoded},
                {"type": "response.output_audio.done"},
                {"type": "response.done"},
            ]
        )

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            events: list[Any] = []
            async for event in connection.events():
                events.append(event)
            deltas = [e for e in events if e.kind == ProviderEventKind.AUDIO_DELTA]
            self.assertEqual(len(deltas), 1)
            self.assertEqual(deltas[0].payload["audio"], pcm)
            self.assertEqual(deltas[0].payload["byte_count"], len(pcm))
            done = [e for e in events if e.kind == ProviderEventKind.AUDIO_DONE]
            self.assertEqual(len(done), 1)
        finally:
            await connection.close()

    async def test_input_transcript_events_normalized(self) -> None:
        socket = FakeLemonadeSocket(
            events=[
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "Hel",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "Hello world.",
                },
                {"type": "response.done"},
            ]
        )

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            events: list[Any] = []
            async for event in connection.events():
                events.append(event)
            delta_events = [
                e for e in events if e.kind == ProviderEventKind.INPUT_TRANSCRIPT_DELTA
            ]
            final_events = [
                e for e in events if e.kind == ProviderEventKind.INPUT_TRANSCRIPT_FINAL
            ]
            self.assertEqual(delta_events[0].payload["delta"], "Hel")
            self.assertEqual(final_events[0].payload["text"], "Hello world.")
        finally:
            await connection.close()

    async def test_error_event_surfaced_as_provider_error(self) -> None:
        socket = FakeLemonadeSocket(
            events=[
                {"type": "error", "error": {"message": "bridge crashed"}},
                {"type": "response.done"},
            ]
        )

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            events: list[Any] = []
            async for event in connection.events():
                events.append(event)
            errors = [e for e in events if e.kind == ProviderEventKind.ERROR]
            self.assertEqual(len(errors), 1)
            self.assertIn("bridge crashed", errors[0].payload["message"])
        finally:
            await connection.close()

    async def test_request_response_sends_response_create(self) -> None:
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            socket.sent.clear()
            await connection.request_response(instructions="Be brief.")
            self.assertEqual(
                socket.sent[-1],
                {
                    "type": "response.create",
                    "response": {"instructions": "Be brief."},
                },
            )
        finally:
            await connection.close()

    async def test_clear_and_cancel_send_correct_events(self) -> None:
        socket = FakeLemonadeSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeLemonadeSocket:
            return socket

        provider = LemonadeRealtimeAgentProvider(socket_factory=_factory)
        with clean_lemonade_env():
            connection = await provider.connect(_make_config())
        try:
            socket.sent.clear()
            await connection.clear_audio()
            await connection.cancel_response()
            self.assertEqual(socket.sent[0], {"type": "input_audio_buffer.clear"})
            self.assertEqual(socket.sent[1], {"type": "response.cancel"})
        finally:
            await connection.close()


if __name__ == "__main__":
    unittest.main()

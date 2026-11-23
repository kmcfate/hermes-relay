"""Tests for the Unmute provider-native realtime-agent adapter.

Stdlib ``unittest`` + a fake WebSocket factory. Mirrors
``test_realtime_agent_xai_provider.py`` with unmute-specific quirks:

* auth header is ``kyutai-api-key`` (default ``public_token``), with an
  optional ``Authorization: Bearer`` for the orchestrator's LLM client
* session.update uses ``voice`` at the top level (not just
  ``audio.output.voice``) and omits the OpenAI-only
  ``output_modalities`` field
* session.instructions is sent as a typed object (smalltalk/constant) per
  the unmute orchestrator's Instructions union, and allow_recording is
  a required boolean field
* response.audio.delta events carry base64 PCM, which the connection
  class decodes back to bytes before yielding to the broker

The orchestrator's ``ClientEvent`` union only accepts ``session.update``,
``input_audio_buffer.append``, and
``unmute.input_audio_buffer.append_anonymized``. Every other client
frame is rejected as "Invalid message", so the lifecycle methods that
would normally emit ``response.create`` / ``input_audio_buffer.clear`` /
``response.cancel`` are deliberate no-ops here — the tests assert that
no-op behaviour so a future refactor cannot silently break the session.
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
from plugin.relay.realtime_agent.providers.unmute import (
    DEFAULT_PUBLIC_TOKEN,
    DEFAULT_URL,
    UnmuteRealtimeAgentProvider,
)
from plugin.voice_lab.providers.base import ProviderUnavailable


class FakeUnmuteSocket:
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
def clean_unmute_env():
    """Drop every ``UNMUTE_*`` env var so URL/auth resolution is deterministic."""
    with patch.dict(os.environ):
        for key in [k for k in os.environ if k.startswith("UNMUTE_")]:
            os.environ.pop(key, None)
        yield


def _make_config(**overrides: Any) -> RealtimeAgentSessionConfig:
    base: dict[str, Any] = {
        "provider": "unmute_realtime",
        "model": "moshi-realtime",
        "voice": "Watercooler",
        "sample_rate": 24000,
        "profile": "default",
        "hermes_session_id": None,
        "instructions": "",
    }
    base.update(overrides)
    return RealtimeAgentSessionConfig(**base)


class UnmuteRealtimeAgentProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_id_and_capabilities(self) -> None:
        async def _factory(*_: Any) -> FakeUnmuteSocket:
            return FakeUnmuteSocket()

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        self.assertEqual(provider.provider_id, "unmute_realtime")
        caps = provider.capabilities
        self.assertEqual(caps.provider_id, "unmute_realtime")
        self.assertTrue(caps.supports_function_calls)
        self.assertTrue(caps.supports_server_vad)

    async def test_default_url_and_public_token_auth(self) -> None:
        seen: dict[str, Any] = {}
        socket = FakeUnmuteSocket(events=[{"type": "response.done"}])

        async def _factory(
            url: str, headers: dict[str, str], timeout: float
        ) -> FakeUnmuteSocket:
            seen["url"] = url
            seen["headers"] = headers
            seen["timeout"] = timeout
            return socket

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        with clean_unmute_env():
            connection = await provider.connect(_make_config())

        self.assertEqual(seen["url"], DEFAULT_URL)
        self.assertEqual(seen["headers"]["kyutai-api-key"], DEFAULT_PUBLIC_TOKEN)
        self.assertNotIn("Authorization", seen["headers"])
        self.assertIsNotNone(connection)
        await connection.close()
        self.assertTrue(socket.closed)

    async def test_uses_explicit_provider_url(self) -> None:
        seen: dict[str, Any] = {}

        async def _factory(
            url: str, headers: dict[str, str], timeout: float
        ) -> FakeUnmuteSocket:
            seen["url"] = url
            return FakeUnmuteSocket(events=[{"type": "response.done"}])

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        config = _make_config()
        config.provider_options["url"] = "ws://unmute.example.test:8000/v1/realtime"
        with clean_unmute_env():
            await provider.connect(config)
        self.assertEqual(seen["url"], "ws://unmute.example.test:8000/v1/realtime")

    async def test_env_url_used_when_no_provider_option(self) -> None:
        seen: dict[str, Any] = {}

        async def _factory(
            url: str, headers: dict[str, str], timeout: float
        ) -> FakeUnmuteSocket:
            seen["url"] = url
            return FakeUnmuteSocket(events=[{"type": "response.done"}])

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        with clean_unmute_env():
            os.environ["UNMUTE_REALTIME_URL"] = (
                "ws://unmute.example.test:9000/v1/realtime"
            )
            await provider.connect(_make_config())
        self.assertEqual(seen["url"], "ws://unmute.example.test:9000/v1/realtime")

    async def test_provider_option_url_wins_over_env(self) -> None:
        seen: dict[str, Any] = {}

        async def _factory(
            url: str, headers: dict[str, str], timeout: float
        ) -> FakeUnmuteSocket:
            seen["url"] = url
            return FakeUnmuteSocket(events=[{"type": "response.done"}])

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        config = _make_config()
        config.provider_options["url"] = "ws://option.example.test:8000/v1/realtime"
        with clean_unmute_env():
            os.environ["UNMUTE_REALTIME_URL"] = (
                "ws://env.example.test:9000/v1/realtime"
            )
            await provider.connect(config)
        self.assertEqual(seen["url"], "ws://option.example.test:8000/v1/realtime")

    async def test_unsupported_sample_rate_rejected(self) -> None:
        async def _factory(*_: Any) -> FakeUnmuteSocket:
            return FakeUnmuteSocket()

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        with clean_unmute_env():
            with self.assertRaisesRegex(ProviderUnavailable, "sample_rate=24000"):
                await provider.connect(_make_config(sample_rate=16000))

    async def test_session_update_uses_unmute_shape(self) -> None:
        socket = FakeUnmuteSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeUnmuteSocket:
            return socket

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        with clean_unmute_env():
            connection = await provider.connect(_make_config(voice="Watercooler"))
        try:
            self.assertTrue(socket.sent, "expected session.update to be sent")
            session_update = socket.sent[0]
            self.assertEqual(session_update["type"], "session.update")
            session = session_update["session"]
            self.assertEqual(session["voice"], "Watercooler")
            self.assertEqual(session["audio"]["output"]["voice"], "Watercooler")
            self.assertEqual(session["audio"]["input"]["format"]["rate"], 24000)
            self.assertIsNone(session["audio"]["input"]["turn_detection"])
            # Unmute-specific fields:
            self.assertIn("allow_recording", session)
            self.assertIsInstance(session["allow_recording"], bool)
            self.assertIsInstance(session["instructions"], dict)
            self.assertIn("type", session["instructions"])
            self.assertEqual(session["instructions"]["type"], "smalltalk")
            # OpenAI-only knobs the orchestrator does not accept:
            self.assertNotIn("output_modalities", session)
        finally:
            await connection.close()

    async def test_session_update_honors_explicit_instructions_object(self) -> None:
        socket = FakeUnmuteSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeUnmuteSocket:
            return socket

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        config = _make_config()
        config.provider_options["instructions"] = {
            "type": "constant",
            "text": "Speak like a pirate.",
            "language": "en",
        }
        config.provider_options["allow_recording"] = True
        with clean_unmute_env():
            connection = await provider.connect(config)
        try:
            session = socket.sent[0]["session"]
            self.assertEqual(session["instructions"]["type"], "constant")
            self.assertEqual(session["instructions"]["text"], "Speak like a pirate.")
            self.assertTrue(session["allow_recording"])
        finally:
            await connection.close()

    async def test_session_update_wraps_string_instructions_as_constant(self) -> None:
        socket = FakeUnmuteSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeUnmuteSocket:
            return socket

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        config = _make_config()
        config.provider_options["instructions"] = "Speak like a pirate."
        with clean_unmute_env():
            connection = await provider.connect(config)
        try:
            session = socket.sent[0]["session"]
            self.assertEqual(session["instructions"]["type"], "constant")
            self.assertEqual(session["instructions"]["text"], "Speak like a pirate.")
        finally:
            await connection.close()

    async def test_audio_delta_event_decoded_to_pcm(self) -> None:
        pcm = b"\xff\xfe\xfd\xfc"
        encoded = base64.b64encode(pcm).decode("ascii")
        socket = FakeUnmuteSocket(
            events=[
                {"type": "response.audio.delta", "delta": encoded},
                {"type": "response.audio.done"},
                {"type": "response.done"},
            ]
        )

        async def _factory(*_: Any) -> FakeUnmuteSocket:
            return socket

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        with clean_unmute_env():
            connection = await provider.connect(_make_config())
        try:
            events: list[Any] = []
            async for event in connection.events():
                events.append(event)
            audio_deltas = [e for e in events if e.kind == ProviderEventKind.AUDIO_DELTA]
            self.assertEqual(len(audio_deltas), 1)
            self.assertEqual(audio_deltas[0].payload["audio"], pcm)
            self.assertEqual(audio_deltas[0].payload["byte_count"], len(pcm))
            self.assertEqual(audio_deltas[0].payload["audio_base64"], encoded)
        finally:
            await connection.close()

    async def test_input_transcript_events_normalized(self) -> None:
        socket = FakeUnmuteSocket(
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

        async def _factory(*_: Any) -> FakeUnmuteSocket:
            return socket

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        with clean_unmute_env():
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

    async def test_function_call_event_emitted_once(self) -> None:
        socket = FakeUnmuteSocket(
            events=[
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "call_1",
                    "name": "hermes_run_task",
                    "arguments": json.dumps({"text": "what time is it"}),
                },
                {"type": "response.done"},
            ]
        )

        async def _factory(*_: Any) -> FakeUnmuteSocket:
            return socket

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        with clean_unmute_env():
            connection = await provider.connect(_make_config())
        try:
            events: list[Any] = []
            async for event in connection.events():
                events.append(event)
            function_calls = [
                e
                for e in events
                if e.kind == ProviderEventKind.FUNCTION_CALL_COMPLETED
            ]
            self.assertEqual(len(function_calls), 1)
            call = function_calls[0].payload["call"]
            self.assertEqual(call.call_id, "call_1")
            self.assertEqual(call.name, "hermes_run_task")
            self.assertEqual(call.arguments, {"text": "what time is it"})
        finally:
            await connection.close()

    async def test_error_event_surfaced_as_provider_error(self) -> None:
        socket = FakeUnmuteSocket(
            events=[
                {
                    "type": "error",
                    "error": {"message": "LLM upstream timed out"},
                },
                {"type": "response.done"},
            ]
        )

        async def _factory(*_: Any) -> FakeUnmuteSocket:
            return socket

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        with clean_unmute_env():
            connection = await provider.connect(_make_config())
        try:
            events: list[Any] = []
            async for event in connection.events():
                events.append(event)
            errors = [e for e in events if e.kind == ProviderEventKind.ERROR]
            self.assertEqual(len(errors), 1)
            self.assertIn("LLM upstream timed out", errors[0].payload["message"])
        finally:
            await connection.close()

    async def test_bearer_header_attached_when_configured(self) -> None:
        seen: dict[str, Any] = {}

        async def _factory(
            url: str, headers: dict[str, str], timeout: float
        ) -> FakeUnmuteSocket:
            seen["headers"] = headers
            return FakeUnmuteSocket(events=[{"type": "response.done"}])

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        config = _make_config()
        config.provider_options["bearer"] = "test-bearer-token-123"
        with clean_unmute_env():
            await provider.connect(config)
        self.assertEqual(
            seen["headers"]["Authorization"], "Bearer test-bearer-token-123"
        )
        self.assertIn("kyutai-api-key", seen["headers"])

    async def test_request_response_is_a_noop(self) -> None:
        """``response.create`` is not in the orchestrator's ClientEvent union."""
        socket = FakeUnmuteSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeUnmuteSocket:
            return socket

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        with clean_unmute_env():
            connection = await provider.connect(_make_config())
        try:
            socket.sent.clear()
            await connection.request_response(
                instructions="Speak in iambic pentameter."
            )
            self.assertEqual(socket.sent, [])
        finally:
            await connection.close()

    async def test_lifecycle_methods_are_noops(self) -> None:
        """clear / cancel / commit / text / tool-result frames are rejected."""
        socket = FakeUnmuteSocket(events=[{"type": "response.done"}])

        async def _factory(*_: Any) -> FakeUnmuteSocket:
            return socket

        provider = UnmuteRealtimeAgentProvider(socket_factory=_factory)
        with clean_unmute_env():
            connection = await provider.connect(_make_config())
        try:
            socket.sent.clear()
            await connection.clear_audio()
            await connection.cancel_response()
            await connection.commit_audio()
            await connection.send_text("hello")
            await connection.send_tool_result("call_1", {"answer": "done"})
            await connection.append_context_item(role="user", text="hello")
            self.assertEqual(socket.sent, [])
        finally:
            await connection.close()


if __name__ == "__main__":
    unittest.main()

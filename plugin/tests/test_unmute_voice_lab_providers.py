"""Tests for the Unmute (moshi-server) plugin TTS + STT providers.

Exercises ``plugin.unmute_provider`` end-to-end against an in-process
fake WebSocket that simulates the moshi-server MessagePack wire
protocol. The fakes mirror the frame shapes documented in
``moshi-server/src/asr.rs`` and ``moshi-server/src/tts.rs``.
"""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import msgpack

from plugin.unmute_provider import (
    DEFAULT_FRAME_SAMPLES,
    DEFAULT_SAMPLE_RATE,
    UnmuteEndpoint,
    UnmuteSTTProvider,
    UnmuteTTSProvider,
    _resolve_unmute_config,
)


# ---------------------------------------------------------------------------
# Fake WebSocket
# ---------------------------------------------------------------------------


class _FakeMessagePackSocket:
    """Synchronous WebSocket fake that emits MessagePack frames.

    moshi-server speaks MessagePack binary frames over the wire; the
    relay's WebSocket-client library treats incoming frames as either
    str or bytes depending on the opcode. We accept both shapes from
    the caller and serialize outgoing ``send()`` calls as bytes so the
    provider's ``msgpack.packb(...)`` round-trip is exercised exactly
    as it is in production.
    """

    def __init__(self, outgoing: list[dict[str, Any]]) -> None:
        self._outgoing = list(outgoing)
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.captured_url: str | None = None
        self.captured_headers: list[str] | None = None
        self.captured_timeout: float | None = None

    def send(self, payload: Any) -> None:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.sent.append(msgpack.unpackb(payload, raw=False))

    def recv(self) -> bytes:
        if not self._outgoing:
            # Return an empty bytes frame to keep the recv loop's
            # ``if raw in (None, b"", "")`` continue-branch quiet.
            return b""
        return msgpack.packb(self._outgoing.pop(0))

    def close(self) -> None:
        self.closed = True


def _make_socket_factory(socket: _FakeMessagePackSocket):
    def factory(url: str, headers: list[str], timeout: float) -> _FakeMessagePackSocket:
        socket.captured_url = url
        socket.captured_headers = headers
        socket.captured_timeout = timeout
        return socket

    return factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_sine_wave(path: Path, duration_seconds: float = 0.4) -> None:
    """Write a 24 kHz mono int16 WAV with a 440 Hz sine wave for STT tests."""
    sample_rate = DEFAULT_SAMPLE_RATE
    amplitude = 8000
    total_frames = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        import math

        for i in range(total_frames):
            value = int(amplitude * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            frames.extend(struct.pack("<h", value))
        wav.writeframes(bytes(frames))


def _float32_chunk(amplitude: float = 0.1, count: int = DEFAULT_FRAME_SAMPLES) -> bytes:
    return struct.pack(f"<{count}f", *([amplitude] * count))


# ---------------------------------------------------------------------------
# STT tests
# ---------------------------------------------------------------------------


class UnmuteSTTProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = UnmuteEndpoint(
            stt_url="ws://stt.example.test:8091/api/chat",
            timeout_seconds=2.0,
        )
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.audio_path = Path(self.tmpdir.name) / "tone.wav"
        _write_sine_wave(self.audio_path)

    def test_default_registry_lists_unmute_stt(self) -> None:
        from plugin.voice_lab.registry import default_registry

        ids = default_registry().provider_ids()
        self.assertIn("unmute_stt", ids)
        self.assertIn("unmute_tts", ids)

    def test_unmute_stt_transcribes_words_from_messagepack_stream(self) -> None:
        # Simulate the server pushing "hello world" as two OutMsg::Word
        # frames plus a final EndWord + Step so the provider exits cleanly.
        socket = _FakeMessagePackSocket(
            [
                {"type": "Ready"},
                {"type": "Word", "text": "hello", "start_time": 0.05},
                {"type": "EndWord", "stop_time": 0.30},
                {"type": "Word", "text": "world", "start_time": 0.31},
                {"type": "EndWord", "stop_time": 0.65},
                {"type": "Step", "step_idx": 32, "prs": [0.1], "buffered_pcm": 0},
            ]
        )
        provider = UnmuteSTTProvider(
            endpoint=self.endpoint,
            socket_factory=_make_socket_factory(socket),
        )
        result = provider.transcribe(str(self.audio_path))

        self.assertEqual(result["success"], True)
        self.assertEqual(result["transcript"], "hello world")
        self.assertEqual(result["provider"], "unmute")
        self.assertGreaterEqual(result["metadata"]["frames_sent"], 1)
        self.assertEqual(result["metadata"]["endpoint"], self.endpoint.stt_url)
        # The provider must have sent an Init frame, then one or more Audio
        # frames, then entered the recv loop. We don't assert exact Audio
        # frame count because the WAV is short; just verify the protocol.
        msg_types = [m["type"] for m in socket.sent]
        self.assertEqual(msg_types[0], "Init")
        self.assertGreater(len([t for t in msg_types if t == "Audio"]), 0)
        self.assertTrue(socket.closed)

    def test_unmute_stt_propagates_server_error_envelope(self) -> None:
        socket = _FakeMessagePackSocket(
            [{"type": "Error", "message": "ASR model not warmed up"}]
        )
        provider = UnmuteSTTProvider(
            endpoint=self.endpoint,
            socket_factory=_make_socket_factory(socket),
        )
        result = provider.transcribe(str(self.audio_path))

        self.assertEqual(result["success"], False)
        self.assertIn("ASR model not warmed up", result["error"])
        self.assertEqual(result["provider"], "unmute")
        self.assertTrue(socket.closed)

    def test_unmute_stt_handles_missing_audio_file(self) -> None:
        provider = UnmuteSTTProvider(
            endpoint=self.endpoint,
            socket_factory=lambda *a, **kw: _FakeMessagePackSocket([]),
        )
        result = provider.transcribe(str(Path("/nope/missing.wav")))
        self.assertEqual(result["success"], False)
        self.assertIn("could not decode", result["error"])

    def test_unmute_stt_returns_voice_lab_transcription_response(self) -> None:
        from plugin.voice_lab.metrics import MetricsRecorder
        from plugin.voice_lab.providers.base import VoiceTranscriptionRequest
        from plugin.voice_lab.providers.unmute_stt import UnmuteVoiceLabSTTProvider

        socket = _FakeMessagePackSocket(
            [
                {"type": "Word", "text": "ready", "start_time": 0.1},
                {"type": "EndWord", "stop_time": 0.5},
                {"type": "Step", "step_idx": 16, "prs": [0.1], "buffered_pcm": 0},
            ]
        )
        provider = UnmuteVoiceLabSTTProvider(
            endpoint=self.endpoint,
            socket_factory=_make_socket_factory(socket),
        )
        recorder = MetricsRecorder()
        response = provider.run_transcription(
            VoiceTranscriptionRequest(
                audio_path=self.audio_path,
                provider_options={},
            ),
            recorder,
        )
        self.assertEqual(response.provider, "unmute_stt")
        self.assertEqual(response.transcript, "ready")
        self.assertGreaterEqual(response.metrics.response_done_ms, 0.0)

    def test_unmute_stt_voicelab_run_text_rejects(self) -> None:
        from plugin.voice_lab.metrics import MetricsRecorder
        from plugin.voice_lab.providers.base import (
            ProviderUnavailable,
            VoiceRequest,
        )
        from plugin.voice_lab.providers.unmute_stt import UnmuteVoiceLabSTTProvider

        provider = UnmuteVoiceLabSTTProvider(endpoint=self.endpoint)
        recorder = MetricsRecorder()
        with self.assertRaisesRegex(ProviderUnavailable, "STT provider"):
            provider.run_text(
                VoiceRequest(
                    text="hello",
                    expression=type("E", (), {"to_dict": lambda self: {}})(),
                    output_path=Path(self.tmpdir.name) / "x.wav",
                ),
                recorder,
            )


# ---------------------------------------------------------------------------
# TTS tests
# ---------------------------------------------------------------------------


class UnmuteTTSProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = UnmuteEndpoint(
            tts_url="ws://stt.example.test:8092/api/tts",
            timeout_seconds=2.0,
        )
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_unmute_tts_synthesizes_wav_from_messagepack_audio_chunks(self) -> None:
        # Simulate the server pushing two OutMsg::Audio chunks followed by
        # an explicit close so the provider exits the recv loop. moshi-server
        # doesn't emit a "done" marker on TTS streams so the provider relies
        # on its own timeout — the test injects a tight 2s timeout.
        chunk1 = _float32_chunk(amplitude=0.10, count=DEFAULT_FRAME_SAMPLES)
        chunk2 = _float32_chunk(amplitude=0.05, count=DEFAULT_FRAME_SAMPLES)
        socket = _FakeMessagePackSocket(
            [
                {"type": "Ready"},
                {"type": "Audio", "pcm": list(struct.unpack(f"<{DEFAULT_FRAME_SAMPLES}f", chunk1))},
                {"type": "Audio", "pcm": list(struct.unpack(f"<{DEFAULT_FRAME_SAMPLES}f", chunk2))},
            ]
        )
        # Close the socket after two recv calls so the provider returns
        # rather than waiting for the timeout — keeps the test fast.
        original_recv = socket.recv

        def recv_then_close():
            try:
                return original_recv()
            except IndexError:
                socket.close()
                return b""

        socket.recv = recv_then_close  # type: ignore[assignment]

        provider = UnmuteTTSProvider(
            endpoint=self.endpoint,
            socket_factory=_make_socket_factory(socket),
        )
        output_path = Path(self.tmpdir.name) / "out.wav"
        result_path = provider.synthesize(
            text="hello world",
            output_path=str(output_path),
            voice="default",
        )
        self.assertEqual(result_path, str(output_path))
        # The provider wrote a real WAV file with the right shape.
        with wave.open(result_path, "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), DEFAULT_SAMPLE_RATE)
            self.assertEqual(wav.getnframes(), DEFAULT_FRAME_SAMPLES * 2)
        # And it sent a Text request frame first.
        sent_types = [m["type"] for m in socket.sent]
        self.assertEqual(sent_types[0], "Text")
        self.assertEqual(socket.sent[0]["text"], "hello world")
        self.assertEqual(socket.sent[0]["voice"], "default")

    def test_unmute_tts_propagates_server_error(self) -> None:
        socket = _FakeMessagePackSocket(
            [{"type": "Error", "message": "voice not found"}]
        )
        provider = UnmuteTTSProvider(
            endpoint=self.endpoint,
            socket_factory=_make_socket_factory(socket),
        )
        with self.assertRaisesRegex(RuntimeError, "voice not found"):
            provider.synthesize(
                text="hi",
                output_path=str(Path(self.tmpdir.name) / "x.wav"),
            )
        self.assertTrue(socket.closed)

    def test_unmute_tts_raises_when_no_audio_received(self) -> None:
        socket = _FakeMessagePackSocket(
            [{"type": "Ready"}]
        )  # No Audio frames → empty buffer → must raise.

        def recv_then_close():
            try:
                return socket._outgoing.pop(0) and msgpack.packb({})
            except IndexError:
                socket.close()
                return b""

        original = socket.recv
        socket.recv = lambda: (  # type: ignore[assignment]
            original() if socket._outgoing else (socket.close(), b"")[1]
        )
        provider = UnmuteTTSProvider(
            endpoint=self.endpoint,
            socket_factory=_make_socket_factory(socket),
        )
        with self.assertRaisesRegex(RuntimeError, "no audio bytes"):
            provider.synthesize(
                text="hi",
                output_path=str(Path(self.tmpdir.name) / "x.wav"),
            )

    def test_unmute_tts_voicelab_run_text_returns_voice_response(self) -> None:
        from plugin.voice_lab.expressions import VoiceExpression
        from plugin.voice_lab.metrics import MetricsRecorder
        from plugin.voice_lab.providers.base import VoiceRequest
        from plugin.voice_lab.providers.unmute_tts import UnmuteVoiceLabTTSProvider

        chunk = _float32_chunk(amplitude=0.10, count=DEFAULT_FRAME_SAMPLES)
        socket = _FakeMessagePackSocket(
            [
                {"type": "Audio", "pcm": list(struct.unpack(f"<{DEFAULT_FRAME_SAMPLES}f", chunk))},
            ]
        )
        original_recv = socket.recv

        def recv_then_close():
            try:
                return original_recv()
            except IndexError:
                socket.close()
                return b""

        socket.recv = recv_then_close  # type: ignore[assignment]

        provider = UnmuteVoiceLabTTSProvider(
            endpoint=self.endpoint,
            socket_factory=_make_socket_factory(socket),
        )
        recorder = MetricsRecorder()
        output_path = Path(self.tmpdir.name) / "lab.wav"
        response = provider.run_text(
            VoiceRequest(
                text="hi",
                expression=VoiceExpression(),
                output_path=output_path,
                provider_options={},
            ),
            recorder,
        )
        self.assertEqual(response.provider, "unmute_tts")
        self.assertEqual(response.model, "moshi-tts")
        self.assertTrue(output_path.exists())
        self.assertEqual(response.metadata["format"], "wav")


# ---------------------------------------------------------------------------
# Config resolution tests
# ---------------------------------------------------------------------------


class UnmuteConfigResolutionTests(unittest.TestCase):
    def test_resolve_unmute_config_prefers_config_over_env(self) -> None:
        class _Config:
            class relay:
                unmute = type(
                    "U",
                    (),
                    {
                        "stt_url": "ws://config.example.test:9001/api/chat",
                        "tts_url": "ws://config.example.test:9002/api/tts",
                        "timeout_seconds": 33.0,
                        "api_key": "config-key",
                    },
                )()

        with patch.dict(
            "os.environ",
            {
                "UNMUTE_STT_URL": "ws://env.example.test:8091/api/chat",
                "UNMUTE_TTS_URL": "ws://env.example.test:8092/api/tts",
                "UNMUTE_API_KEY": "env-key",
            },
            clear=False,
        ):
            cfg = _resolve_unmute_config(_Config())
        self.assertEqual(cfg.stt_url, "ws://config.example.test:9001/api/chat")
        self.assertEqual(cfg.tts_url, "ws://config.example.test:9002/api/tts")
        self.assertEqual(cfg.api_key, "config-key")
        self.assertEqual(cfg.timeout_seconds, 33.0)

    def test_resolve_unmute_config_falls_back_to_env_when_config_missing(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "UNMUTE_STT_URL": "ws://env.example.test:8091/api/chat",
                "UNMUTE_TTS_URL": "ws://env.example.test:8092/api/tts",
                "UNMUTE_API_KEY": "env-key",
            },
            clear=False,
        ):
            cfg = _resolve_unmute_config(None)
        self.assertEqual(cfg.stt_url, "ws://env.example.test:8091/api/chat")
        self.assertEqual(cfg.tts_url, "ws://env.example.test:8092/api/tts")
        self.assertEqual(cfg.api_key, "env-key")


if __name__ == "__main__":
    unittest.main()
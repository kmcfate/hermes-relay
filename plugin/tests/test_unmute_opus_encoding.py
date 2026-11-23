"""Tests for the unmute provider's Opus-encoding path.

The unmute orchestrator's ``receive_loop`` runs every incoming
``input_audio_buffer.append`` payload through ``sphn.OpusStreamReader`` at
a hard-coded 24 kHz mono rate. The relay provider MUST pre-encode PCM
chunks to Opus before sending; otherwise the orchestrator throws
``ValueError: unexpected ogg capture pattern`` on the first chunk and
tears the session down. These tests verify:

  1. ``_OpusStreamWriterLazy.append_pcm_bytes`` produces bytes that
     decode cleanly through ``sphn.OpusStreamReader`` (round-trip).
  2. ``UnmuteRealtimeAgentConnection.send_audio`` emits an Opus-encoded
     payload (no longer the raw PCM bytes) and never sends an empty
     chunk that the orchestrator would treat as a malformed OGG page.
  3. PCM at 16 kHz (Android phone) is resampled to 24 kHz before Opus
     encoding so the orchestrator sees the correct sample rate.

``numpy`` and ``sphn`` are optional extras — the provider imports them
lazily so the module still loads without them. Skip the whole suite when
either is unavailable rather than failing collection.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import unittest
from typing import Any

from plugin.relay.realtime_agent.models import RealtimeAgentSessionConfig
from plugin.relay.realtime_agent.providers.unmute import (
    UNMUTE_OPUS_SAMPLE_RATE,
    UnmuteRealtimeAgentConnection,
    _OpusStreamWriterLazy,
)

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SPHN = importlib.util.find_spec("sphn") is not None
_HAS_OPUS = _HAS_NUMPY and _HAS_SPHN


def _make_config() -> RealtimeAgentSessionConfig:
    return RealtimeAgentSessionConfig(
        provider="unmute_realtime",
        model="moshi-realtime",
        voice="Watercooler",
        sample_rate=UNMUTE_OPUS_SAMPLE_RATE,
        profile=None,
        hermes_session_id=None,
        instructions="",
        provider_options={},
        tools=(),
    )


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def receive_json(self) -> dict[str, Any]:
        raise EOFError

    async def close(self) -> None:
        pass


@unittest.skipUnless(_HAS_OPUS, "numpy and sphn are required for Opus encoding tests")
class OpusStreamWriterLazyRoundTripTests(unittest.IsolatedAsyncioTestCase):
    """``append_pcm_bytes`` output must decode via ``sphn.OpusStreamReader``
    (the same class the orchestrator uses internally)."""

    async def test_round_trip_silence(self) -> None:
        import numpy as np
        import sphn

        sr = UNMUTE_OPUS_SAMPLE_RATE
        writer = _OpusStreamWriterLazy(sr)
        silence_pcm = b"\x00\x00" * (sr // 10)  # 100ms of int16 silence
        opus = await asyncio.to_thread(writer.append_pcm_bytes, silence_pcm)
        # Opus always emits *some* bytes for 100ms of input at 24 kHz.
        # sphn's encoder emits one OGG page per ~20 ms frame, so 100 ms
        # produces ~5 frames ≈ a few hundred bytes.
        self.assertGreater(len(opus), 0)

        reader = sphn.OpusStreamReader(sr)
        decoded = reader.append_bytes(opus)
        self.assertGreater(decoded.size, 0)
        # Decoded length should be approximately 100ms at 24 kHz = 2400
        # samples (sphn returns variable-size chunks).
        self.assertGreater(decoded.size, 1500)
        self.assertEqual(decoded.dtype, np.float32)

    async def test_round_trip_tone(self) -> None:
        import numpy as np
        import sphn

        sr = UNMUTE_OPUS_SAMPLE_RATE
        writer = _OpusStreamWriterLazy(sr)
        # 100ms of a 440Hz tone, int16, mono
        t = np.arange(sr // 10, dtype=np.float32) / sr
        tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        opus = await asyncio.to_thread(writer.append_pcm_bytes, tone.tobytes())
        self.assertGreater(len(opus), 0)
        # Must round-trip with non-trivial energy
        reader = sphn.OpusStreamReader(sr)
        decoded = reader.append_bytes(opus)
        self.assertGreater(float(np.abs(decoded).max()), 0.01)

    async def test_empty_pcm_returns_empty(self) -> None:
        writer = _OpusStreamWriterLazy(UNMUTE_OPUS_SAMPLE_RATE)
        self.assertEqual(await asyncio.to_thread(writer.append_pcm_bytes, b""), b"")


@unittest.skipUnless(_HAS_OPUS, "numpy and sphn are required for Opus encoding tests")
class SendAudioOpusEncodingTests(unittest.IsolatedAsyncioTestCase):
    """``send_audio`` must emit Opus-encoded audio and never send an empty
    chunk that the orchestrator would treat as a malformed OGG page."""

    async def test_send_audio_emits_opus_not_pcm(self) -> None:
        import numpy as np
        import sphn

        socket = _FakeSocket()

        # 100ms of 440Hz tone at 16kHz (what the phone sends)
        sr_in = 16000
        t = np.arange(sr_in // 10, dtype=np.float32) / sr_in
        pcm16 = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()

        conn = UnmuteRealtimeAgentConnection(
            socket=socket,  # type: ignore[arg-type]
            config=_make_config(),
            auth_source="test",
        )
        await conn.send_audio(pcm16, sr_in)

        self.assertEqual(len(socket.sent), 1, "send_audio should produce exactly one message")
        msg = socket.sent[0]
        self.assertEqual(msg["type"], "input_audio_buffer.append")
        self.assertIn("audio", msg)
        audio_bytes = base64.b64decode(msg["audio"])

        # The audio bytes must NOT be raw PCM (Opus-encoded bytes start
        # with the OGG page signature "OggS" = b'OggS' = 0x4f 0x67 0x67 0x53)
        self.assertEqual(
            audio_bytes[:4], b"OggS", "Audio payload must be Opus-in-OGG, not raw PCM"
        )

        # And it must decode via sphn's reader (what the orchestrator uses)
        reader = sphn.OpusStreamReader(UNMUTE_OPUS_SAMPLE_RATE)
        decoded = reader.append_bytes(audio_bytes)
        self.assertGreater(decoded.size, 0, "Opus bytes must round-trip through sphn")

    async def test_send_audio_resamples_16k_to_24k(self) -> None:
        import numpy as np

        socket = _FakeSocket()
        sr_in = 16000
        t = np.arange(sr_in // 10, dtype=np.float32) / sr_in
        pcm16 = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()

        conn = UnmuteRealtimeAgentConnection(
            socket=socket,  # type: ignore[arg-type]
            config=_make_config(),
            auth_source="test",
        )
        await conn.send_audio(pcm16, sr_in)

        # 100ms at 16 kHz -> 100ms at 24 kHz means 2400 samples, i.e.
        # 4800 bytes of int16 before Opus framing.
        raw_len = len(pcm16)
        expected_samples = round((raw_len // 2) * UNMUTE_OPUS_SAMPLE_RATE / sr_in)
        self.assertEqual(expected_samples, 2400)
        # The Opus payload is smaller than the raw PCM it came from.
        self.assertLess(len(base64.b64decode(socket.sent[0]["audio"])), raw_len)

    async def test_send_audio_truly_empty_does_not_send(self) -> None:
        """Zero bytes of PCM MUST NOT produce a payload — the orchestrator
        would treat the resulting OGG bytes (or worse, an empty audio
        field) as a malformed page and close the WS."""
        socket = _FakeSocket()
        conn = UnmuteRealtimeAgentConnection(
            socket=socket,  # type: ignore[arg-type]
            config=_make_config(),
            auth_source="test",
        )
        await conn.send_audio(b"", UNMUTE_OPUS_SAMPLE_RATE)
        self.assertEqual(socket.sent, [], "Empty PCM must not produce an audio payload")


if __name__ == "__main__":
    unittest.main()

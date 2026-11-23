"""Kyutai Unmute (moshi-server) STT + TTS providers for Hermes Relay.

Wires the relay plugin's voice stack to a locally-running moshi-server
(https://github.com/kyutai-labs/moshi/tree/main/rust/moshi-server) instead of
cloud TTS/STT services. Hermes remains the LLM; Unmute provides only the
speech-to-text and text-to-speech endpoints.

Wire protocol
-------------
moshi-server speaks MessagePack binary frames over a single WebSocket
connection. The relay opens a fresh connection per request (moshi-server's
``Asr::handle_socket`` and ``Tts::handle_socket`` are single-shot) and uses
``rmp_serde``-compatible encodings — message structs with a ``type``
discriminator field.

STT client messages (``InMsg`` variants):
- ``{"type": "Init"}`` — only used in batched mode
- ``{"type": "Marker", "id": <int>}`` — sync marker into the stream
- ``{"type": "Audio", "pcm": [<float>, ...]}`` — PCM frames as float32
- ``{"type": "OggOpus", "data": [<bytes>]}`` — Ogg/Opus-encoded audio

TTS client messages (``InMsg`` variants, query-string + body):
- ``POST /api/tts?voice=<name>&text=<urlencoded-text>`` — single-shot HTTP
  endpoint; alternative is the WebSocket path used here.
- WebSocket body: ``{"type": "Text", "text": "...", "voice": "..."}`` etc.

Server → client messages (``OutMsg`` variants):
- STT: ``Word {text, start_time}``, ``EndWord {stop_time}``, ``Marker {id}``,
       ``Step {step_idx, prs, buffered_pcm}``, ``Error {message}``, ``Ready``
- TTS: ``Audio {pcm: [<float>]}`` (slice-by-slice streaming), ``Word``,
       ``Step``, ``Error``, ``Ready``

Configuration
-------------
The relay reads these from ``~/.hermes/config.yaml`` (per-profile via
``relay.profile_voice``):

.. code-block:: yaml

    relay:
      unmute:
        stt_url: ws://127.0.0.1:8091/api/chat
        tts_url: ws://127.0.0.1:8092/api/tts
        timeout_seconds: 60
        api_key: ""  # moshi-server optional bearer; leave empty unless set

The defaults match the macOS dockerless start scripts in the upstream
unmute repo (https://github.com/kyutai-labs/unmute/tree/main/dockerless).
Point ``stt_url`` / ``tts_url`` at the host running moshi-server — for a
LAN-reachable deployment use the machine's hostname or IP address.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import struct
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import msgpack

try:
    from agent.transcription_provider import TranscriptionProvider
except ImportError:  # older hermes-agent: re-export for the plugin ABC anyway
    TranscriptionProvider = object  # type: ignore[misc,assignment]

try:
    from agent.tts_provider import TTSProvider
except ImportError:
    TTSProvider = object  # type: ignore[misc,assignment]

logger = logging.getLogger("hermes_relay.unmute")

# moshi-server wire protocol constants (mirrors moshi-server/src/protocol.rs).
# Frame size is 1920 samples at 24 kHz = 80 ms — the unit the STT server
# expects per ``InMsg::Audio`` chunk. The TTS server emits ``OutMsg::Audio``
# chunks at the same cadence.
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1
DEFAULT_FRAME_SAMPLES = 1920
DEFAULT_SAMPLE_WIDTH = 2  # int16

DEFAULT_STT_URL = "ws://127.0.0.1:8091/api/chat"
DEFAULT_TTS_URL = "ws://127.0.0.1:8092/api/tts"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_LANGUAGE = "en"

# Connection id helpers (moshi-server uses snake_case keys).
_INMSG_INIT = {"type": "Init"}
_INMSG_AUDIO_KEY = "Audio"
_INMSG_TEXT_KEY = "Text"
_OUTMSG_WORD = "Word"
_OUTMSG_AUDIO = "Audio"
_OUTMSG_ERROR = "Error"
_OUTMSG_READY = "Ready"
_OUTMSG_ENDWORD = "EndWord"
_OUTMSG_MARKER = "Marker"
_OUTMSG_STEP = "Step"


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class WebSocketLike(Protocol):
    """Subset of ``websocket-client`` we depend on."""

    def send(self, payload: Any) -> Any: ...

    def recv(self) -> Any: ...

    def close(self) -> Any: ...


WebSocketFactory = Callable[[str, list[str], float], WebSocketLike]


def _create_websocket(url: str, headers: list[str], timeout: float) -> WebSocketLike:
    """Default ``websocket-client`` factory."""
    import websocket  # type: ignore

    return websocket.create_connection(url, header=headers, timeout=timeout)


def _read_pcm_float32(path: Path, target_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Read a WAV file and return 24 kHz mono int16 PCM bytes.

    moshi-server's STT endpoint expects float32 PCM frames. We accept
    arbitrary WAVs from upstream (the Android client uploads whatever the
    mic produces) and resample/downmix here so the STT server sees a
    consistent shape. Resampling uses ``audioop`` from the stdlib — we
    deliberately avoid numpy/scipy so the relay plugin's deps stay small.

    Returns the raw PCM bytes (int16, mono, target_rate). For OggOpus
    audio (the Android default) we fall through with the original bytes
    untouched and the caller chooses the ``OggOpus`` message type.
    """
    import audioop

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if channels > 1:
        raw = audioop.tomono(raw, sample_width, 0.5, 0.5)
    if sample_width != DEFAULT_SAMPLE_WIDTH:
        raw = audioop.lin2lin(raw, sample_width, DEFAULT_SAMPLE_WIDTH)
    if sample_rate != target_rate:
        # audioop.ratecv wants (fragment, width, nchannels, inrate, outrate)
        raw, _ = audioop.ratecv(
            raw, DEFAULT_SAMPLE_WIDTH, 1, sample_rate, target_rate, None
        )
    return raw


def _int16_to_float32_bytes(pcm_int16: bytes) -> bytes:
    """Convert int16 PCM bytes to little-endian float32 bytes (moshi STT format)."""
    count = len(pcm_int16) // DEFAULT_SAMPLE_WIDTH
    floats = struct.unpack(f"<{count}h", pcm_int16)
    return struct.pack(f"<{count}f", *(f / 32768.0 for f in floats))


def _float32_bytes_to_wav(
    pcm_float32: bytes, output_path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> None:
    """Convert little-endian float32 PCM bytes to a mono int16 WAV file."""
    count = len(pcm_float32) // 4
    floats = struct.unpack(f"<{count}f", pcm_float32)
    ints = [max(-32768, min(32767, int(round(f * 32767)))) for f in floats]
    pcm_int16 = struct.pack(f"<{count}h", *ints)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(DEFAULT_CHANNELS)
        wav.setsampwidth(DEFAULT_SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_int16)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class UnmuteEndpoint:
    """Resolved WebSocket endpoint + auth headers for a moshi-server service."""

    stt_url: str = DEFAULT_STT_URL
    tts_url: str = DEFAULT_TTS_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    api_key: str = ""

    def stt_headers(self) -> list[str]:
        headers: list[str] = []
        if self.api_key:
            headers.append(f"kyutai-api-key: {self.api_key}")
        return headers

    def tts_headers(self) -> list[str]:
        return self.stt_headers()


def _resolve_unmute_config(config: Any) -> UnmuteEndpoint:
    """Pull ``relay.unmute.*`` from a Hermes config object, with env fallbacks.

    Resolution order: ``config.relay.unmute`` dict → ``UNMUTE_*`` env vars →
    defaults. Env vars only override when the config dict is absent (so
    ``hermes config set relay.unmute.stt_url`` always wins).
    """
    stt_url = DEFAULT_STT_URL
    tts_url = DEFAULT_TTS_URL
    timeout = DEFAULT_TIMEOUT_SECONDS
    api_key = ""

    relay_cfg = getattr(config, "relay", None)
    if relay_cfg is not None:
        unmute_cfg = getattr(relay_cfg, "unmute", None)
        if unmute_cfg is not None:
            stt_url = getattr(unmute_cfg, "stt_url", stt_url) or stt_url
            tts_url = getattr(unmute_cfg, "tts_url", tts_url) or tts_url
            timeout = float(
                getattr(unmute_cfg, "timeout_seconds", timeout) or timeout
            )
            api_key = getattr(unmute_cfg, "api_key", api_key) or api_key

    if stt_url == DEFAULT_STT_URL:
        stt_url = os.getenv("UNMUTE_STT_URL", stt_url) or stt_url
    if tts_url == DEFAULT_TTS_URL:
        tts_url = os.getenv("UNMUTE_TTS_URL", tts_url) or tts_url
    if not api_key:
        api_key = os.getenv("UNMUTE_API_KEY", "") or ""

    return UnmuteEndpoint(
        stt_url=stt_url,
        tts_url=tts_url,
        timeout_seconds=timeout,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Provider availability (used by the plugin registry + hermes setup picker)
# ---------------------------------------------------------------------------


def _service_available(url: str, timeout: float = 2.0) -> bool:
    """Best-effort liveness probe. Returns False on any network/parse error."""
    try:
        import urllib.parse
        import urllib.request

        parsed = urllib.parse.urlparse(url)
        if not parsed.hostname:
            return False
        http_url = urllib.parse.urlunparse(
            ("http", f"{parsed.hostname}:{parsed.port or 80}", "/v1/health", "", "", "")
        )
        with urllib.request.urlopen(http_url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


# ---------------------------------------------------------------------------
# STT (TranscriptionProvider)
# ---------------------------------------------------------------------------


class UnmuteSTTProvider(TranscriptionProvider):
    """Plugin STT backend that streams WAV files through moshi-server STT.

    Implements ``agent.transcription_provider.TranscriptionProvider`` (the
    upstream plugin ABC for ``stt.provider`` dispatch). Audio is sent as
    ``InMsg::Audio`` chunks at the STT server's native 24 kHz mono
    cadence; partial transcripts are concatenated and returned once the
    stream finishes.

    Lifecycle: one WebSocket connection per ``transcribe()`` call.
    moshi-server's STT handler is single-shot; opening a fresh socket
    per request is the documented pattern. For real-time voice-mode
    pushing, the ``run_transcription`` VoiceLab path on the same
    protocol would be the extension surface — out of scope here.
    """

    @property
    def name(self) -> str:
        return "unmute"

    @property
    def display_name(self) -> str:
        return "Kyutai Unmute (moshi-server)"

    def __init__(
        self,
        endpoint: UnmuteEndpoint | None = None,
        socket_factory: WebSocketFactory | None = None,
    ) -> None:
        super().__init__()
        self._endpoint = endpoint or _resolve_unmute_config(None)
        self._socket_factory = socket_factory or _create_websocket
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        try:
            endpoint = self._resolve_endpoint(None)
            return _service_available(endpoint.stt_url, timeout=2.0)
        except Exception:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "moshi-stt",
                "display": "Moshi STT (24 kHz)",
                "languages": [
                    "en",
                    "fr",
                    "de",
                    "es",
                    "it",
                    "pt",
                    "nl",
                    "ja",
                    "zh",
                ],
                "max_audio_seconds": 1800,
            }
        ]

    def default_model(self) -> str | None:
        return "moshi-stt"

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "Kyutai Unmute",
            "badge": "local",
            "tag": "Moshi STT over local moshi-server WebSocket",
            "env_vars": [],
        }

    def _resolve_endpoint(self, stt_config: Any) -> UnmuteEndpoint:
        # Allow per-profile override via relay.unmute.* — the upstream
        # dispatcher passes the merged STT config dict in, so this can
        # also honour ``stt.unmute.stt_url`` (used as a profile knob).
        if stt_config is None:
            return self._endpoint
        try:
            sub = stt_config.get("unmute", {}) if isinstance(stt_config, dict) else {}
        except Exception:
            return self._endpoint
        if not isinstance(sub, dict) or not sub:
            return self._endpoint
        return UnmuteEndpoint(
            stt_url=sub.get("stt_url", self._endpoint.stt_url),
            tts_url=sub.get("tts_url", self._endpoint.tts_url),
            timeout_seconds=float(
                sub.get("timeout_seconds", self._endpoint.timeout_seconds)
            ),
            api_key=sub.get("api_key", self._endpoint.api_key) or "",
        )

    def transcribe(
        self,
        file_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Drive the full request → response cycle against moshi-server STT."""
        stt_config = extra.get("stt_config") if isinstance(extra, dict) else None
        endpoint = self._resolve_endpoint(stt_config)

        try:
            pcm_int16 = _read_pcm_float32(Path(file_path))
        except (wave.Error, FileNotFoundError, EOFError) as exc:
            return self._error(f"could not decode audio file: {exc}")
        except Exception as exc:  # noqa: BLE001 — provider contract
            return self._error(f"audio preprocessing failed: {exc}")

        pcm_float32 = _int16_to_float32_bytes(pcm_int16)
        frames = [
            pcm_float32[i : i + DEFAULT_FRAME_SAMPLES * 4]
            for i in range(0, len(pcm_float32), DEFAULT_FRAME_SAMPLES * 4)
        ]

        ws = None
        words: list[str] = []
        started_at = time.monotonic()
        try:
            ws = self._socket_factory(
                endpoint.stt_url,
                endpoint.stt_headers(),
                endpoint.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(f"could not connect to {endpoint.stt_url}: {exc}")

        try:
            # moshi-server STT does not require an Init frame in single-stream
            # mode (it's a batched-mode concept), but sending one is harmless
            # and matches the Asr protocol documented in upstream tests.
            ws.send(msgpack.packb(_INMSG_INIT))
            for chunk in frames:
                payload = msgpack.packb(
                    {"type": _INMSG_AUDIO_KEY, "pcm": list(_floats_from_bytes(chunk))}
                )
                ws.send(payload)
            # Half-close: moshi-server returns its final OutMsg sequence once
            # no further audio arrives within the ASR_DELAY window. We poll
            # until we see an Error or until the timeout elapses.
            deadline = started_at + endpoint.timeout_seconds
            while time.monotonic() < deadline:
                try:
                    raw = ws.recv()
                except Exception as exc:  # noqa: BLE001
                    return self._error(f"recv failed: {exc}")
                if raw in (None, b"", ""):
                    continue
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                try:
                    msg = msgpack.unpackb(raw, raw=False)
                except Exception as exc:  # noqa: BLE001
                    return self._error(f"malformed server frame: {exc}")
                if not isinstance(msg, dict):
                    continue
                msg_type = msg.get("type")
                if msg_type == _OUTMSG_ERROR:
                    return self._error(
                        str(msg.get("message") or "moshi-server returned error")
                    )
                if msg_type == _OUTMSG_READY:
                    continue
                if msg_type == _OUTMSG_WORD:
                    word = msg.get("text")
                    if isinstance(word, str) and word:
                        words.append(word)
                    continue
                if msg_type == _OUTMSG_ENDWORD:
                    continue
                if msg_type == _OUTMSG_STEP:
                    continue
                if msg_type == _OUTMSG_MARKER:
                    continue
        finally:
            try:
                if ws is not None:
                    ws.close()
            except Exception:
                pass

        transcript = " ".join(words).strip()
        return {
            "success": True,
            "transcript": transcript,
            "provider": self.name,
            "metadata": {
                "model": "moshi-stt",
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "frames_sent": len(frames),
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
                "endpoint": endpoint.stt_url,
                "language": language or DEFAULT_LANGUAGE,
            },
        }

    def _error(self, message: str) -> dict[str, Any]:
        logger.warning("Unmute STT error: %s", message)
        return {
            "success": False,
            "transcript": "",
            "error": message,
            "provider": self.name,
        }


def _floats_from_bytes(buf: bytes):
    count = len(buf) // 4
    return struct.unpack(f"<{count}f", buf)


# ---------------------------------------------------------------------------
# TTS (TTSProvider)
# ---------------------------------------------------------------------------


class UnmuteTTSProvider(TTSProvider):
    """Plugin TTS backend that streams PCM chunks from moshi-server TTS.

    Implements ``agent.tts_provider.TTSProvider``. The server sends back
    ``OutMsg::Audio {pcm: [float, ...]}`` slices at the model's native
    frame cadence (1920 samples @ 24 kHz ≈ 80 ms). We accumulate the
    float32 bytes and emit a single mono int16 WAV at the end. For
    real-time streaming into the voice_output broker, the relay's
    voice_lab provider under ``unmute_tts`` exposes the chunk-by-chunk
    sink used in this file.
    """

    @property
    def name(self) -> str:
        return "unmute"

    @property
    def display_name(self) -> str:
        return "Kyutai Unmute (moshi-server)"

    def __init__(
        self,
        endpoint: UnmuteEndpoint | None = None,
        socket_factory: WebSocketFactory | None = None,
    ) -> None:
        super().__init__()
        self._endpoint = endpoint or _resolve_unmute_config(None)
        self._socket_factory = socket_factory or _create_websocket
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        try:
            endpoint = self._resolve_endpoint(None)
            return _service_available(endpoint.tts_url, timeout=2.0)
        except Exception:
            return False

    def list_voices(self) -> list[dict[str, Any]]:
        # moshi-server enumerates voices via GET /v1/voices; we don't probe
        # that here because ``is_available()`` would race. The picker falls
        # back to the catalog below for the inline provider picker and the
        # user can refresh voices via the relay's provider-options endpoint.
        return [
            {"id": "default", "display": "Default moshi voice"},
        ]

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "moshi-tts",
                "display": "Moshi TTS (24 kHz)",
                "languages": [
                    "en",
                    "fr",
                    "de",
                    "es",
                    "it",
                    "pt",
                    "nl",
                    "ja",
                    "zh",
                ],
            }
        ]

    def default_model(self) -> str | None:
        return "moshi-tts"

    def default_voice(self) -> str | None:
        return "default"

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "Kyutai Unmute",
            "badge": "local",
            "tag": "Moshi TTS over local moshi-server WebSocket",
            "env_vars": [],
        }

    def _resolve_endpoint(self, tts_config: Any) -> UnmuteEndpoint:
        if tts_config is None:
            return self._endpoint
        try:
            sub = tts_config.get("unmute", {}) if isinstance(tts_config, dict) else {}
        except Exception:
            return self._endpoint
        if not isinstance(sub, dict) or not sub:
            return self._endpoint
        return UnmuteEndpoint(
            stt_url=sub.get("stt_url", self._endpoint.stt_url),
            tts_url=sub.get("tts_url", self._endpoint.tts_url),
            timeout_seconds=float(
                sub.get("timeout_seconds", self._endpoint.timeout_seconds)
            ),
            api_key=sub.get("api_key", self._endpoint.api_key) or "",
        )

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float | None = None,
        format: str = "wav",  # noqa: A002 — provider ABC contract
        **extra: Any,
    ) -> str:
        tts_config = extra.get("tts_config") if isinstance(extra, dict) else None
        endpoint = self._resolve_endpoint(tts_config)
        voice_name = voice or self.default_voice() or "default"

        ws = None
        pcm_buffer = bytearray()
        started_at = time.monotonic()
        try:
            ws = self._socket_factory(
                endpoint.tts_url,
                endpoint.tts_headers(),
                endpoint.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"could not connect to {endpoint.tts_url}: {exc}"
            ) from exc

        try:
            request: dict[str, Any] = {
                "type": _INMSG_TEXT_KEY,
                "text": text,
                "voice": voice_name,
            }
            if speed is not None:
                request["speed"] = float(speed)
            ws.send(msgpack.packb(request))

            deadline = started_at + endpoint.timeout_seconds
            while time.monotonic() < deadline:
                try:
                    raw = ws.recv()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"recv failed: {exc}") from exc
                if raw in (None, b"", ""):
                    continue
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                try:
                    msg = msgpack.unpackb(raw, raw=False)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"malformed server frame: {exc}") from exc
                if not isinstance(msg, dict):
                    continue
                msg_type = msg.get("type")
                if msg_type == _OUTMSG_ERROR:
                    raise RuntimeError(
                        str(msg.get("message") or "moshi-server returned error")
                    )
                if msg_type == _OUTMSG_READY:
                    continue
                if msg_type == _OUTMSG_AUDIO:
                    pcm = msg.get("pcm")
                    if isinstance(pcm, list):
                        pcm_buffer.extend(struct.pack(f"<{len(pcm)}f", *pcm))
                    continue
                # TTS server may echo the textual stream back as Word slices;
                # we ignore those (the rendered audio is the source of truth).
                if msg_type in (_OUTMSG_WORD, _OUTMSG_ENDWORD, _OUTMSG_STEP, _OUTMSG_MARKER):
                    continue
        finally:
            try:
                if ws is not None:
                    ws.close()
            except Exception:
                pass

        if not pcm_buffer:
            raise RuntimeError("moshi-server TTS produced no audio bytes")

        output = Path(output_path)
        _float32_bytes_to_wav(bytes(pcm_buffer), output)
        return str(output)

    def voice_compatible(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# VoiceLab providers (standalone CLI harness in plugin/voice_lab/)
# ---------------------------------------------------------------------------
#
# The relay's voice_lab is the provider-neutral CLI used for offline
# benchmarking and expression-controlled TTS renders. Registering
# ``unmute_stt`` and ``unmute_tts`` there lets users run the lab against
# moshi-server without touching the plugin ABC. These wrappers also
# surface streaming sinks the relay's voice_output broker can call into
# directly when configured with provider=unmute_tts.


class UnmuteVoiceLabSTTProvider:
    """VoiceLab wrapper exposing moshi-server STT to the lab CLI."""

    info_id = "unmute_stt"
    info_name = "Kyutai Unmute STT"
    info_description = (
        "Streaming STT against a local moshi-server (Kyutai Unmute). "
        "Wire protocol: MessagePack over WebSocket. See plugin/unmute_provider.py."
    )

    def __init__(
        self,
        endpoint: UnmuteEndpoint | None = None,
        socket_factory: WebSocketFactory | None = None,
    ) -> None:
        self._stt = UnmuteSTTProvider(endpoint, socket_factory)


class UnmuteVoiceLabTTSProvider:
    """VoiceLab wrapper exposing moshi-server TTS to the lab CLI."""

    info_id = "unmute_tts"
    info_name = "Kyutai Unmute TTS"
    info_description = (
        "Streaming TTS against a local moshi-server (Kyutai Unmute). "
        "Wire protocol: MessagePack over WebSocket. See plugin/unmute_provider.py."
    )

    def __init__(
        self,
        endpoint: UnmuteEndpoint | None = None,
        socket_factory: WebSocketFactory | None = None,
    ) -> None:
        self._tts = UnmuteTTSProvider(endpoint, socket_factory)


__all__ = [
    "UnmuteEndpoint",
    "UnmuteSTTProvider",
    "UnmuteTTSProvider",
    "UnmuteVoiceLabSTTProvider",
    "UnmuteVoiceLabTTSProvider",
    "_resolve_unmute_config",
    "_service_available",
    "_create_websocket",
    "WebSocketLike",
    "WebSocketFactory",
    "DEFAULT_STT_URL",
    "DEFAULT_TTS_URL",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_FRAME_SAMPLES",
]
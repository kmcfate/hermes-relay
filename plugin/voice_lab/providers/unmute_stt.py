"""VoiceLab wrapper around the Unmute (moshi-server) STT provider.

Bridges the standalone ``plugin.voice_lab`` CLI surface (which expects a
:class:`plugin.voice_lab.providers.base.VoiceProvider`) to the
:class:`agent.transcription_provider.TranscriptionProvider` we ship in
``plugin.unmute_provider``. The voice_lab registry exposes both
``unmute_stt`` (this file) and ``unmute_tts`` (see ``unmute_tts.py``)
so the CLI can benchmark moshi-server without touching the upstream
plugin ABC.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...unmute_provider import (
    DEFAULT_SAMPLE_RATE,
    WebSocketFactory,
    UnmuteEndpoint,
    UnmuteSTTProvider,
)
from .base import (
    ProviderInfo,
    ProviderRunError,
    ProviderUnavailable,
    VoiceProvider,
    VoiceTranscriptionRequest,
    VoiceTranscriptionResponse,
)
from ..metrics import MetricsRecorder

logger = logging.getLogger("hermes_relay.voice_lab.unmute_stt")


@dataclass(slots=True)
class UnmuteSTTVoiceLabAdapter:
    """Adapts the upstream plugin STT ABC into the voice_lab interface.

    The voice_lab's :class:`VoiceTranscriptionRequest` has no
    ``provider_options`` field for STT — only :class:`VoiceRequest` does.
    So this adapter keeps an internal :class:`UnmuteEndpoint` and lets
    the lab user override per-request via the request's
    ``provider_options["stt_url"]`` / ``["api_key"]`` keys when present.
    """

    endpoint: UnmuteEndpoint
    socket_factory: WebSocketFactory | None = None


class UnmuteVoiceLabSTTProvider(VoiceProvider):
    """Local moshi-server STT exposed via the voice_lab CLI."""

    info = ProviderInfo(
        id="unmute_stt",
        name="Kyutai Unmute STT",
        description=(
            "Streaming speech-to-text against a local moshi-server "
            "(Kyutai Unmute). Wire protocol: MessagePack over WebSocket."
        ),
        models=("moshi-stt",),
        languages=("en", "fr", "de", "es", "it", "pt", "nl", "ja", "zh"),
        sample_rates=(DEFAULT_SAMPLE_RATE,),
        supports_stt=True,
        supports_tts=False,
        supports_speech_to_speech=False,
        supports_realtime=False,
        supports_interruption=False,
        supports_expression=False,
    )

    def __init__(
        self,
        endpoint: UnmuteEndpoint | None = None,
        socket_factory: WebSocketFactory | None = None,
    ) -> None:
        self._endpoint = endpoint or UnmuteEndpoint()
        self._socket_factory = socket_factory
        self._delegate: UnmuteSTTProvider | None = None

    def _delegate_for(self, request: VoiceTranscriptionRequest) -> UnmuteSTTProvider:
        # Honor per-request endpoint overrides. The lab CLI does not expose
        # these today, but tests and programmatic users can pass them.
        options = request.provider_options or {}
        if "stt_url" in options or "api_key" in options:
            endpoint = UnmuteEndpoint(
                stt_url=options.get("stt_url", self._endpoint.stt_url),
                tts_url=options.get("tts_url", self._endpoint.tts_url),
                timeout_seconds=float(
                    options.get("timeout_seconds", self._endpoint.timeout_seconds)
                ),
                api_key=options.get("api_key", self._endpoint.api_key) or "",
            )
        else:
            endpoint = self._endpoint
        return UnmuteSTTProvider(
            endpoint=endpoint, socket_factory=self._socket_factory
        )

    def run_text(self, request, recorder: MetricsRecorder):  # type: ignore[override]
        raise ProviderUnavailable(
            "unmute_stt is an STT provider; run_transcription() is its surface"
        )

    def run_transcription(
        self,
        request: VoiceTranscriptionRequest,
        recorder: MetricsRecorder,
    ) -> VoiceTranscriptionResponse:
        recorder.event(
            "request_started",
            provider=self.info.id,
            model="moshi-stt",
            audio_path=str(request.audio_path),
            expected_text=request.expected_text,
        )
        delegate = self._delegate_for(request)
        result = delegate.transcribe(
            str(request.audio_path),
            language=request.provider_options.get("language"),
        )
        if not result.get("success"):
            recorder.event("transcript_failed", error=result.get("error"))
            raise ProviderRunError(
                result.get("error") or "unmute STT returned no transcript"
            )
        transcript = result.get("transcript") or ""
        recorder.event(
            "transcript_done",
            text_chars=len(transcript),
            endpoint=self._endpoint.stt_url,
        )
        metrics = recorder.finish(provider=self.info.id, model="moshi-stt")
        return VoiceTranscriptionResponse(
            provider=self.info.id,
            model="moshi-stt",
            transcript=transcript,
            audio_path=request.audio_path,
            metrics=metrics,
            metadata=result.get("metadata") or {},
        )

    def run_speech_to_speech(self, request, recorder: MetricsRecorder):  # type: ignore[override]
        raise ProviderUnavailable(
            "unmute_stt does not support speech-to-speech; use unmute_tts "
            "with a separate STT pass, or wire the unmute /v1/realtime "
            "endpoint (not yet shipped)."
        )


__all__ = ["UnmuteVoiceLabSTTProvider", "UnmuteSTTVoiceLabAdapter"]
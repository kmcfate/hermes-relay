"""VoiceLab wrapper around the Unmute (moshi-server) TTS provider.

Bridges the standalone ``plugin.voice_lab`` CLI surface to
:class:`agent.tts_provider.TTSProvider` (the upstream plugin ABC). Pairs
with ``unmute_stt.py`` so the lab CLI can benchmark both halves of the
Kyutai Unmute pipeline without touching the upstream plugin ABC.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ...unmute_provider import (
    DEFAULT_SAMPLE_RATE,
    WebSocketFactory,
    UnmuteEndpoint,
    UnmuteTTSProvider,
)
from .base import (
    ProviderInfo,
    ProviderRunError,
    ProviderUnavailable,
    VoiceProvider,
    VoiceRequest,
    VoiceResponse,
)
from ..metrics import MetricsRecorder

logger = logging.getLogger("hermes_relay.voice_lab.unmute_tts")


class UnmuteVoiceLabTTSProvider(VoiceProvider):
    """Local moshi-server TTS exposed via the voice_lab CLI."""

    info = ProviderInfo(
        id="unmute_tts",
        name="Kyutai Unmute TTS",
        description=(
            "Streaming text-to-speech against a local moshi-server "
            "(Kyutai Unmute). Wire protocol: MessagePack over WebSocket."
        ),
        models=("moshi-tts",),
        voices=("default",),
        languages=("en", "fr", "de", "es", "it", "pt", "nl", "ja", "zh"),
        sample_rates=(DEFAULT_SAMPLE_RATE,),
        supports_tts=True,
        supports_stt=False,
        supports_speech_to_speech=False,
        supports_realtime=False,
        supports_interruption=False,
        supports_expression=True,
    )

    def __init__(
        self,
        endpoint: UnmuteEndpoint | None = None,
        socket_factory: WebSocketFactory | None = None,
    ) -> None:
        self._endpoint = endpoint or UnmuteEndpoint()
        self._socket_factory = socket_factory
        self._delegate: UnmuteTTSProvider | None = None

    def _delegate_for(self, request: VoiceRequest) -> UnmuteTTSProvider:
        options = request.provider_options or {}
        if any(k in options for k in ("tts_url", "api_key")):
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
        return UnmuteTTSProvider(
            endpoint=endpoint, socket_factory=self._socket_factory
        )

    def run_text(
        self,
        request: VoiceRequest,
        recorder: MetricsRecorder,
    ) -> VoiceResponse:
        recorder.event(
            "request_started",
            provider=self.info.id,
            model="moshi-tts",
            text_chars=len(request.text),
            expression=request.expression.to_dict(),
        )
        delegate = self._delegate_for(request)
        try:
            output_path = delegate.synthesize(
                text=request.text,
                output_path=str(request.output_path),
                voice=request.provider_options.get("voice"),
                speed=request.provider_options.get("speed"),
            )
        except RuntimeError as exc:
            recorder.event("synth_failed", error=str(exc))
            raise ProviderRunError(str(exc)) from exc
        metrics = recorder.finish(provider=self.info.id, model="moshi-tts")
        return VoiceResponse(
            provider=self.info.id,
            model="moshi-tts",
            voice=request.provider_options.get("voice") or "default",
            audio_path=request.output_path,
            metrics=metrics,
            metadata={
                "endpoint": self._endpoint.tts_url,
                "format": "wav",
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "output_path": output_path,
            },
        )

    def run_transcription(
        self,
        request,
        recorder: MetricsRecorder,
    ):
        raise ProviderUnavailable(
            "unmute_tts is a TTS provider; use unmute_stt for transcription"
        )

    def run_speech_to_speech(
        self,
        request,
        recorder: MetricsRecorder,
    ):
        raise ProviderUnavailable(
            "unmute_tts does not support speech-to-speech; use a separate "
            "STT pass + TTS synthesis, or wire the unmute /v1/realtime "
            "endpoint (not yet shipped)."
        )


__all__ = ["UnmuteVoiceLabTTSProvider"]
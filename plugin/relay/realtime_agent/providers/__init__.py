"""Provider adapters for the relay realtime-agent broker."""

from __future__ import annotations

from .base import RealtimeAgentProviderAdapter
from .openai import OpenAIRealtimeAgentAdapter, OpenAIRealtimeAgentProvider
from .lemonade import LemonadeRealtimeAgentAdapter, LemonadeRealtimeAgentProvider
from .unmute import UnmuteRealtimeAgentAdapter, UnmuteRealtimeAgentProvider
from .xai import XAIRealtimeAgentAdapter, XAIRealtimeAgentProvider


def adapter_for(provider_id: str) -> RealtimeAgentProviderAdapter:
    if provider_id == OpenAIRealtimeAgentAdapter.provider_id:
        return OpenAIRealtimeAgentAdapter()
    if provider_id == XAIRealtimeAgentAdapter.provider_id:
        return XAIRealtimeAgentAdapter()
    if provider_id == UnmuteRealtimeAgentAdapter.provider_id:
        return UnmuteRealtimeAgentAdapter()
    if provider_id == LemonadeRealtimeAgentAdapter.provider_id:
        return LemonadeRealtimeAgentAdapter()
    return RealtimeAgentProviderAdapter(provider_id)


__all__ = [
    "LemonadeRealtimeAgentAdapter",
    "LemonadeRealtimeAgentProvider",
    "OpenAIRealtimeAgentAdapter",
    "OpenAIRealtimeAgentProvider",
    "RealtimeAgentProviderAdapter",
    "UnmuteRealtimeAgentAdapter",
    "UnmuteRealtimeAgentProvider",
    "XAIRealtimeAgentAdapter",
    "XAIRealtimeAgentProvider",
    "adapter_for",
]

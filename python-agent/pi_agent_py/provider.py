from __future__ import annotations

from typing import AsyncIterator, Protocol

from .types import ProviderContext, ProviderEvent


class LLMProvider(Protocol):
    async def stream(self, context: ProviderContext) -> AsyncIterator[ProviderEvent]:
        ...

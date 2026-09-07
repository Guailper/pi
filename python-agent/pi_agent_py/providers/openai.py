from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import AsyncIterator

from ..types import Message, ProviderContext, ProviderEvent, ToolCall


class OpenAIChatProvider:
    """Optional adapter for the OpenAI Python SDK Chat Completions streaming API."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError('Install optional dependency: pip install -e .[openai]') from exc
        self._client = AsyncOpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'), base_url=base_url)

    @staticmethod
    def _messages(context: ProviderContext) -> list[dict]:
        result: list[dict] = []
        for message in context.messages:
            if message.role == 'user':
                result.append({'role': 'user', 'content': message.content})
            elif message.role == 'assistant':
                payload: dict = {'role': 'assistant', 'content': message.content or None}
                if message.tool_calls:
                    payload['tool_calls'] = [
                        {
                            'id': call.id,
                            'type': 'function',
                            'function': {'name': call.name, 'arguments': json.dumps(call.arguments, ensure_ascii=False)},
                        }
                        for call in message.tool_calls
                    ]
                result.append(payload)
            elif message.role == 'tool_result':
                result.append({'role': 'tool', 'tool_call_id': message.tool_call_id, 'content': message.content})
        return result

    async def stream(self, context: ProviderContext) -> AsyncIterator[ProviderEvent]:
        tools = [
            {'type': 'function', 'function': {'name': t.name, 'description': t.description, 'parameters': t.parameters}}
            for t in context.tools
        ]
        stream = await self._client.chat.completions.create(
            model=context.model,
            messages=self._messages(context),
            tools=tools or None,
            stream=True,
        )
        text_parts: list[str] = []
        tool_fragments: dict[int, dict[str, str]] = defaultdict(lambda: {'id': '', 'name': '', 'arguments': ''})
        finish_reason = None
        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                text_parts.append(delta.content)
                yield ProviderEvent(type='text_delta', text=delta.content)
            for tc in delta.tool_calls or []:
                frag = tool_fragments[tc.index]
                if tc.id:
                    frag['id'] = tc.id
                if tc.function and tc.function.name:
                    frag['name'] += tc.function.name
                if tc.function and tc.function.arguments:
                    frag['arguments'] += tc.function.arguments
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        calls: list[ToolCall] = []
        for idx in sorted(tool_fragments):
            frag = tool_fragments[idx]
            try:
                args = json.loads(frag['arguments'] or '{}')
            except json.JSONDecodeError:
                args = {'_raw_arguments': frag['arguments']}
            call = ToolCall(id=frag['id'] or f'call_{idx}', name=frag['name'], arguments=args)
            calls.append(call)
            yield ProviderEvent(type='tool_call', tool_call=call)

        stop_reason = 'tool_use' if calls else ('length' if finish_reason == 'length' else 'stop')
        yield ProviderEvent(type='done', message=Message(role='assistant', content=''.join(text_parts), tool_calls=calls, stop_reason=stop_reason))

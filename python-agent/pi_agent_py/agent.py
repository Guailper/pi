from __future__ import annotations

import asyncio
import inspect
from collections import deque
from dataclasses import replace
from typing import Any, Awaitable, Callable

from .provider import LLMProvider
from .schema import SchemaError, validate
from .types import (
    AfterToolCall,
    AgentEvent,
    AgentState,
    AgentTool,
    BeforeToolCall,
    Message,
    ProviderContext,
    QueueMode,
    ShouldStopAfterTurn,
    ToolCall,
    ToolContext,
    ToolExecutionMode,
    ToolResult,
    TransformContext,
)

EventListener = Callable[[AgentEvent], Awaitable[None] | None]


class _PendingQueue:
    def __init__(self, mode: QueueMode) -> None:
        self.mode = mode
        self._items: deque[Message] = deque()

    def push(self, message: Message) -> None:
        self._items.append(message)

    def drain(self) -> list[Message]:
        if not self._items:
            return []
        if self.mode == 'all':
            items = list(self._items)
            self._items.clear()
            return items
        return [self._items.popleft()]

    def clear(self) -> None:
        self._items.clear()


class Agent:
    """A small Python reimplementation of Pi's stateful agent runtime."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        system_prompt: str = '',
        tools: list[AgentTool] | None = None,
        thinking_level: str = 'off',
        tool_execution: ToolExecutionMode = 'parallel',
        steering_mode: QueueMode = 'one-at-a-time',
        follow_up_mode: QueueMode = 'one-at-a-time',
        transform_context: TransformContext | None = None,
        before_tool_call: BeforeToolCall | None = None,
        after_tool_call: AfterToolCall | None = None,
        should_stop_after_turn: ShouldStopAfterTurn | None = None,
    ) -> None:
        self.provider = provider
        self.state = AgentState(
            system_prompt=system_prompt,
            model=model,
            tools=list(tools or []),
            thinking_level=thinking_level,
        )
        self.tool_execution = tool_execution
        self.transform_context = transform_context
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.should_stop_after_turn = should_stop_after_turn
        self._listeners: list[EventListener] = []
        self._steering = _PendingQueue(steering_mode)
        self._follow_up = _PendingQueue(follow_up_mode)
        self._abort_event = asyncio.Event()
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        self._running = False

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    async def _emit(self, event_type: str, **data: Any) -> None:
        event = AgentEvent(event_type, data)
        for listener in list(self._listeners):
            result = listener(event)
            if inspect.isawaitable(result):
                await result

    def steer(self, message: str | Message) -> None:
        self._steering.push(self._as_user_message(message))

    def follow_up(self, message: str | Message) -> None:
        self._follow_up.push(self._as_user_message(message))

    def clear_queues(self) -> None:
        self._steering.clear()
        self._follow_up.clear()

    def abort(self) -> None:
        self._abort_event.set()

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()

    def reset(self) -> None:
        if self._running:
            raise RuntimeError('agent is running')
        self.state.messages.clear()
        self.state.error_message = None
        self.state.streaming_message = None
        self.state.pending_tool_calls.clear()
        self.clear_queues()

    async def prompt(self, message: str | Message) -> list[Message]:
        if self._running:
            raise RuntimeError('agent is already processing; use steer() or follow_up()')
        prompt = self._as_user_message(message)
        return await self._run([prompt], continuation=False)

    async def continue_run(self) -> list[Message]:
        if self._running:
            raise RuntimeError('agent is already processing')
        if not self.state.messages:
            raise RuntimeError('no messages to continue from')
        if self.state.messages[-1].role == 'assistant':
            queued = self._steering.drain() or self._follow_up.drain()
            if queued:
                return await self._run(queued, continuation=False)
            raise RuntimeError('cannot continue from assistant message')
        return await self._run([], continuation=True)

    def _as_user_message(self, message: str | Message) -> Message:
        if isinstance(message, Message):
            return message
        return Message(role='user', content=message)

    async def _run(self, prompts: list[Message], *, continuation: bool) -> list[Message]:
        self._running = True
        self._idle_event.clear()
        self._abort_event = asyncio.Event()
        self.state.is_streaming = True
        self.state.error_message = None
        new_messages: list[Message] = []
        try:
            await self._emit('agent_start')
            if prompts:
                await self._emit('turn_start')
                for prompt in prompts:
                    self.state.messages.append(prompt)
                    new_messages.append(prompt)
                    await self._emit('message_start', message=prompt)
                    await self._emit('message_end', message=prompt)
            elif continuation:
                await self._emit('turn_start')

            first_turn_open = True
            pending = self._steering.drain()
            stop_requested = False
            while True:
                has_more_tool_calls = True
                while has_more_tool_calls or pending:
                    if not first_turn_open:
                        await self._emit('turn_start')
                    first_turn_open = False

                    for queued in pending:
                        self.state.messages.append(queued)
                        new_messages.append(queued)
                        await self._emit('message_start', message=queued)
                        await self._emit('message_end', message=queued)
                    pending = []

                    if self._abort_event.is_set():
                        aborted = Message(role='assistant', stop_reason='aborted', error_message='aborted by user')
                        self.state.messages.append(aborted)
                        new_messages.append(aborted)
                        await self._emit('message_start', message=aborted)
                        await self._emit('message_end', message=aborted)
                        await self._emit('turn_end', message=aborted, tool_results=[])
                        break

                    assistant = await self._stream_assistant()
                    new_messages.append(assistant)
                    if assistant.stop_reason in {'error', 'aborted'}:
                        await self._emit('turn_end', message=assistant, tool_results=[])
                        has_more_tool_calls = False
                        break

                    tool_results: list[Message] = []
                    has_more_tool_calls = False
                    if assistant.tool_calls:
                        if assistant.stop_reason == 'length':
                            for call in assistant.tool_calls:
                                result = Message(
                                    role='tool_result',
                                    content='Tool call not executed because model output was truncated.',
                                    tool_call_id=call.id,
                                    tool_name=call.name,
                                    is_error=True,
                                )
                                await self._emit('tool_execution_start', tool_call_id=call.id, tool_name=call.name, args=call.arguments)
                                await self._emit('tool_execution_end', tool_call_id=call.id, tool_name=call.name, result=result, is_error=True)
                                tool_results.append(result)
                        else:
                            tool_results, terminate = await self._execute_tool_batch(assistant)
                            has_more_tool_calls = not terminate
                        for result in tool_results:
                            self.state.messages.append(result)
                            new_messages.append(result)
                            await self._emit('message_start', message=result)
                            await self._emit('message_end', message=result)

                    await self._emit('turn_end', message=assistant, tool_results=tool_results)
                    if self.should_stop_after_turn and await self.should_stop_after_turn(assistant, tool_results, self.state):
                        stop_requested = True
                        has_more_tool_calls = False
                        pending = []
                        break
                    pending = self._steering.drain()

                if stop_requested or self._abort_event.is_set() or (new_messages and new_messages[-1].stop_reason in {'error', 'aborted'}):
                    break
                followups = self._follow_up.drain()
                if followups:
                    pending = followups
                    continue
                break

            await self._emit('agent_end', messages=new_messages)
            return new_messages
        finally:
            self.state.is_streaming = False
            self.state.streaming_message = None
            self.state.pending_tool_calls.clear()
            self._running = False
            self._idle_event.set()

    async def _stream_assistant(self) -> Message:
        messages = list(self.state.messages)
        if self.transform_context:
            messages = await self.transform_context(messages)
        context = ProviderContext(
            system_prompt=self.state.system_prompt,
            messages=messages,
            tools=list(self.state.tools),
            model=self.state.model,
            thinking_level=self.state.thinking_level,
        )
        partial = Message(role='assistant')
        await self._emit('message_start', message=partial)
        self.state.streaming_message = partial
        final: Message | None = None
        try:
            async for event in self.provider.stream(context):
                if self._abort_event.is_set():
                    final = Message(role='assistant', content=partial.content, stop_reason='aborted', error_message='aborted by user')
                    break
                if event.type == 'text_delta':
                    partial.content += event.text
                    self.state.streaming_message = replace(partial)
                    await self._emit('message_update', message=replace(partial), delta=event.text)
                elif event.type == 'tool_call' and event.tool_call:
                    partial.tool_calls.append(event.tool_call)
                    await self._emit('message_update', message=replace(partial), tool_call=event.tool_call)
                elif event.type == 'done' and event.message:
                    final = event.message
                elif event.type == 'error':
                    final = event.message or Message(role='assistant', content=partial.content, stop_reason='error', error_message='provider error')
            if final is None:
                final = Message(
                    role='assistant',
                    content=partial.content,
                    tool_calls=list(partial.tool_calls),
                    stop_reason='tool_use' if partial.tool_calls else 'stop',
                )
        except Exception as exc:
            final = Message(role='assistant', content=partial.content, stop_reason='error', error_message=str(exc))

        self.state.messages.append(final)
        self.state.streaming_message = None
        self.state.error_message = final.error_message
        await self._emit('message_end', message=final)
        return final

    async def _execute_tool_batch(self, assistant: Message) -> tuple[list[Message], bool]:
        prepared: list[tuple[int, ToolCall, AgentTool | None, dict[str, Any] | None, Message | None, bool]] = []
        any_sequential = False
        tools_by_name = {tool.name: tool for tool in self.state.tools}

        for idx, call in enumerate(assistant.tool_calls):
            tool = tools_by_name.get(call.name)
            await self._emit('tool_execution_start', tool_call_id=call.id, tool_name=call.name, args=call.arguments)
            if tool is None:
                prepared.append((idx, call, None, None, self._tool_error(call, f'Unknown tool: {call.name}'), False))
                continue
            try:
                args = validate(tool.parameters, call.arguments)
                if self.before_tool_call:
                    decision = await self.before_tool_call(call, args, self.state)
                    if decision and decision.get('block'):
                        msg = self._tool_error(call, decision.get('reason') or 'Tool call blocked by before_tool_call hook')
                        prepared.append((idx, call, tool, args, msg, bool(decision.get('terminate'))))
                        continue
                any_sequential = any_sequential or tool.execution_mode == 'sequential'
                prepared.append((idx, call, tool, args, None, False))
            except SchemaError as exc:
                prepared.append((idx, call, tool, None, self._tool_error(call, str(exc)), False))

        results: list[tuple[int, Message, bool]] = []
        for idx, call, tool, args, immediate, terminate in prepared:
            if immediate is not None:
                await self._emit('tool_execution_end', tool_call_id=call.id, tool_name=call.name, result=immediate, is_error=True)
                results.append((idx, immediate, terminate))

        executable = [item for item in prepared if item[4] is None and item[2] is not None]
        mode = 'sequential' if any_sequential or self.tool_execution == 'sequential' else 'parallel'
        if mode == 'sequential':
            for item in executable:
                results.append(await self._execute_one(item))
        else:
            tasks = [asyncio.create_task(self._execute_one(item)) for item in executable]
            for task in asyncio.as_completed(tasks):
                results.append(await task)

        results.sort(key=lambda item: item[0])
        messages = [item[1] for item in results]
        terminate = bool(results) and all(item[2] for item in results)
        return messages, terminate

    async def _execute_one(self, item: tuple[int, ToolCall, AgentTool | None, dict[str, Any] | None, Message | None, bool]) -> tuple[int, Message, bool]:
        idx, call, tool, args, _, _ = item
        assert tool is not None and args is not None
        self.state.pending_tool_calls.add(call.id)

        async def update(partial: Any) -> None:
            await self._emit('tool_execution_update', tool_call_id=call.id, tool_name=call.name, partial_result=partial)

        is_error = False
        try:
            result = await tool.execute(call.id, args, ToolContext(self._abort_event, update))
        except Exception as exc:
            is_error = True
            result = ToolResult(content=f'{type(exc).__name__}: {exc}')
        finally:
            self.state.pending_tool_calls.discard(call.id)

        if self.after_tool_call:
            override = await self.after_tool_call(call, result, is_error, self.state)
            if override:
                if 'content' in override:
                    result.content = str(override['content'])
                if 'details' in override:
                    result.details = override['details']
                if 'terminate' in override:
                    result.terminate = bool(override['terminate'])
                if 'is_error' in override:
                    is_error = bool(override['is_error'])

        message = Message(
            role='tool_result',
            content=result.content,
            tool_call_id=call.id,
            tool_name=call.name,
            is_error=is_error,
        )
        await self._emit('tool_execution_end', tool_call_id=call.id, tool_name=call.name, result=message, is_error=is_error)
        return idx, message, result.terminate

    @staticmethod
    def _tool_error(call: ToolCall, text: str) -> Message:
        return Message(role='tool_result', content=text, tool_call_id=call.id, tool_name=call.name, is_error=True)

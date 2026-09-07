from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import time
from typing import Any, Awaitable, Callable, Literal

Role = Literal['user', 'assistant', 'tool_result']
ToolExecutionMode = Literal['parallel', 'sequential']
QueueMode = Literal['all', 'one-at-a-time']
StopReason = Literal['stop', 'tool_use', 'length', 'error', 'aborted']


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ''
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    stop_reason: StopReason | None = None
    error_message: str | None = None
    timestamp: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Message':
        calls = [ToolCall(**call) for call in data.get('tool_calls', [])]
        payload = dict(data)
        payload['tool_calls'] = calls
        return cls(**payload)


@dataclass(slots=True)
class ToolResult:
    content: str
    details: Any = None
    terminate: bool = False


@dataclass(slots=True)
class ToolUpdate:
    content: str
    details: Any = None


ToolHandler = Callable[[str, dict[str, Any], 'ToolContext'], Awaitable[ToolResult]]
ToolUpdateCallback = Callable[[ToolUpdate], Awaitable[None]]


@dataclass(slots=True)
class ToolContext:
    abort_event: Any
    on_update: ToolUpdateCallback | None = None

    async def update(self, content: str, details: Any = None) -> None:
        if self.on_update:
            await self.on_update(ToolUpdate(content=content, details=details))


@dataclass(slots=True)
class AgentTool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: ToolHandler
    label: str | None = None
    execution_mode: ToolExecutionMode | None = None


@dataclass(slots=True)
class AgentEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentState:
    system_prompt: str
    model: str
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    thinking_level: str = 'off'
    is_streaming: bool = False
    streaming_message: Message | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error_message: str | None = None


@dataclass(slots=True)
class ProviderContext:
    system_prompt: str
    messages: list[Message]
    tools: list[AgentTool]
    model: str
    thinking_level: str = 'off'


@dataclass(slots=True)
class ProviderEvent:
    type: Literal['start', 'text_delta', 'tool_call', 'done', 'error']
    text: str = ''
    tool_call: ToolCall | None = None
    message: Message | None = None


BeforeToolCall = Callable[[ToolCall, dict[str, Any], AgentState], Awaitable[dict[str, Any] | None]]
AfterToolCall = Callable[[ToolCall, ToolResult, bool, AgentState], Awaitable[dict[str, Any] | None]]
ShouldStopAfterTurn = Callable[[Message, list[Message], AgentState], Awaitable[bool]]
TransformContext = Callable[[list[Message]], Awaitable[list[Message]]]

# Pi Agent Python Reimplementation

这是一个基于 Pi Agent Harness 核心设计重新实现的 Python 版本，重点复现 `pi-agent-core` 的运行时语义，而不是对 TypeScript 源码逐行翻译。

## 已实现

- Stateful Agent Runtime：`AgentState`、消息历史、运行状态
- Agent Loop：LLM → tool calls → tool results → 下一轮 LLM
- Event Streaming：`agent_start / turn_start / message_* / tool_execution_* / turn_end / agent_end`
- Tool schema 校验与统一错误回传
- 多工具并行执行，以及 per-tool `sequential` 覆盖
- `before_tool_call` / `after_tool_call` hooks
- `steer()` / `follow_up()` 队列，支持 `all` 与 `one-at-a-time`
- `should_stop_after_turn` 优雅停止
- JSON Session 持久化
- Coding Agent 基础工具：`read / write / edit / bash`
- Provider 解耦；附带可选 OpenAI Chat Completions streaming adapter

## 架构

```text
Application / CLI
      |
      v
+-----------------------+
|        Agent          |
| state / queues/events |
+-----------+-----------+
            |
            v
+-----------------------+
|      Agent Loop       |
| model -> tools -> ... |
+-----+-------------+---+
      |             |
      v             v
 LLMProvider     AgentTool[]
      |             |
      v             v
 OpenAI/...    read/write/edit/bash
```

这与 Pi 的关键边界保持一致：Agent 层负责状态、循环、工具和事件；具体模型调用由 provider 注入；具体工具由 application 注入。

## 快速开始

```bash
cd python-agent
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\\Scripts\\activate
pip install -e ".[openai]"
export OPENAI_API_KEY="..."  # Windows PowerShell: $env:OPENAI_API_KEY="..."
pi-agent-py --model <your-model> --workspace .
```

或在代码中：

```python
from pi_agent_py import Agent, coding_tools
from pi_agent_py.providers import OpenAIChatProvider

agent = Agent(
    provider=OpenAIChatProvider(),
    model="<your-model>",
    system_prompt="You are a coding agent.",
    tools=coding_tools("."),
)
await agent.prompt("Read pyproject.toml and summarize it")
```

## 核心调用链

```text
prompt()
  -> agent_start
  -> turn_start
  -> append user message
  -> provider.stream(context)
  -> assistant message
  -> parse tool_calls
  -> validate arguments
  -> before_tool_call
  -> execute tool(s)
  -> after_tool_call
  -> append tool_result messages
  -> turn_end
  -> steering queue?
  -> next LLM turn
  -> follow_up queue?
  -> agent_end
```

## 与原 Pi 的对应关系

| Pi | Python 版本 |
|---|---|
| `packages/agent/src/agent.ts` | `pi_agent_py/agent.py::Agent` |
| `agent-loop.ts` | `Agent._run()` / `_stream_assistant()` / `_execute_tool_batch()` |
| `types.ts` | `pi_agent_py/types.py` |
| `StreamFn` / `pi-ai` boundary | `LLMProvider.stream()` |
| `AgentTool` | `pi_agent_py.types.AgentTool` |
| coding-agent basic tools | `pi_agent_py/builtin_tools.py` |

## 安全说明

`read/write/edit` 被限制在 workspace 内；`bash` 与原 Pi 类似，仍以当前用户权限运行，并不是安全沙箱。处理不可信任务时应放在 Docker、VM 或其他 sandbox 中。

## 测试

```bash
pip install -e ".[test]"
pytest -q
```

## 设计取舍

这个目录是“核心机制复现版”，没有复制 Pi 的完整 TUI、遥测、全部 provider、插件生态和所有 session backend。这样代码规模更适合作为 Agent Runtime 学习、面试讲解和二次开发基础。

## Attribution

Design inspired by the MIT-licensed Pi Agent Harness project. This implementation is independently written in Python and keeps attribution to the upstream project.

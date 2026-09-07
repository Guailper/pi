from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .agent import Agent
from .builtin_tools import coding_tools
from .providers import OpenAIChatProvider
from .session import load_session, save_session


async def main_async() -> None:
    parser = argparse.ArgumentParser(description='Small Python reimplementation of Pi agent runtime')
    parser.add_argument('--model', required=True, help='Model id accepted by your provider')
    parser.add_argument('--workspace', default='.')
    parser.add_argument('--session', default='.pi-python/session.json')
    parser.add_argument('--system', default='You are a coding agent. Inspect before editing, use tools when needed, and keep changes minimal.')
    args = parser.parse_args()

    agent = Agent(
        provider=OpenAIChatProvider(), model=args.model, system_prompt=args.system,
        tools=coding_tools(args.workspace), tool_execution='parallel',
    )
    session = Path(args.session)
    if session.exists():
        agent.state.messages = load_session(session)

    async def print_event(event):
        if event.type == 'message_update' and event.data.get('delta'):
            print(event.data['delta'], end='', flush=True)
        elif event.type == 'tool_execution_start':
            print(f"\n[tool] {event.data['tool_name']} {event.data['args']}")
        elif event.type == 'tool_execution_end':
            print(f"[tool:{'error' if event.data['is_error'] else 'ok'}] {event.data['tool_name']}")

    agent.subscribe(print_event)
    print('pi-agent-py. Commands: /exit /reset')
    while True:
        try:
            text = await asyncio.to_thread(input, '\n> ')
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip() == '/exit':
            break
        if text.strip() == '/reset':
            agent.reset()
            save_session(session, agent.state.messages)
            print('session reset')
            continue
        await agent.prompt(text)
        save_session(session, agent.state.messages)
        print()


def main() -> None:
    asyncio.run(main_async())


if __name__ == '__main__':
    main()

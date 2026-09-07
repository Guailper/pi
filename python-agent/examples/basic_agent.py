import asyncio

from pi_agent_py import Agent, coding_tools
from pi_agent_py.providers import OpenAIChatProvider


async def main():
    agent = Agent(
        provider=OpenAIChatProvider(),
        model='<your-model>',
        system_prompt='You are a concise coding assistant.',
        tools=coding_tools('.'),
    )

    async def events(event):
        if event.type == 'message_update' and event.data.get('delta'):
            print(event.data['delta'], end='', flush=True)

    agent.subscribe(events)
    await agent.prompt('List the files in this project by using bash.')


asyncio.run(main())

import asyncio

from pi_agent_py import Agent, AgentTool, Message, ProviderEvent, ToolCall, ToolResult


class FakeProvider:
    def __init__(self):
        self.turn = 0

    async def stream(self, context):
        self.turn += 1
        if self.turn == 1:
            yield ProviderEvent(type='tool_call', tool_call=ToolCall('1', 'echo', {'text': 'hello'}))
            yield ProviderEvent(type='done', message=Message(role='assistant', tool_calls=[ToolCall('1', 'echo', {'text': 'hello'})], stop_reason='tool_use'))
        else:
            yield ProviderEvent(type='text_delta', text='done')
            yield ProviderEvent(type='done', message=Message(role='assistant', content='done', stop_reason='stop'))


def test_tool_loop():
    async def run():
        async def echo(_id, args, _ctx):
            return ToolResult(args['text'])

        tool = AgentTool('echo', 'echo text', {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}, echo)
        agent = Agent(provider=FakeProvider(), model='fake', tools=[tool])
        await agent.prompt('go')
        assert [m.role for m in agent.state.messages] == ['user', 'assistant', 'tool_result', 'assistant']
        assert agent.state.messages[-2].content == 'hello'
        assert agent.state.messages[-1].content == 'done'

    asyncio.run(run())


def test_schema_error_becomes_tool_result():
    class BadArgs:
        def __init__(self):
            self.turn = 0

        async def stream(self, context):
            self.turn += 1
            if self.turn == 1:
                call = ToolCall('2', 'echo', {})
                yield ProviderEvent(type='done', message=Message(role='assistant', tool_calls=[call], stop_reason='tool_use'))
            else:
                yield ProviderEvent(type='done', message=Message(role='assistant', content='ok', stop_reason='stop'))

    async def run():
        async def echo(_id, args, _ctx):
            return ToolResult('never')
        tool = AgentTool('echo', 'echo', {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}, echo)
        agent = Agent(provider=BadArgs(), model='fake', tools=[tool], should_stop_after_turn=lambda *_: asyncio.sleep(0, result=True))
        await agent.prompt('go')
        result = next(m for m in agent.state.messages if m.role == 'tool_result')
        assert result.is_error
        assert 'required field missing' in result.content

    asyncio.run(run())


def test_event_lifecycle_without_tools():
    class Provider:
        async def stream(self, context):
            yield ProviderEvent(type='text_delta', text='hi')
            yield ProviderEvent(type='done', message=Message(role='assistant', content='hi', stop_reason='stop'))

    async def run():
        events = []
        agent = Agent(provider=Provider(), model='fake')
        agent.subscribe(lambda e: events.append(e.type))
        await agent.prompt('hello')
        assert events == [
            'agent_start', 'turn_start', 'message_start', 'message_end',
            'message_start', 'message_update', 'message_end', 'turn_end', 'agent_end'
        ]

    asyncio.run(run())


def test_parallel_tools_preserve_source_order():
    class Provider:
        def __init__(self):
            self.turn = 0

        async def stream(self, context):
            self.turn += 1
            if self.turn == 1:
                calls = [ToolCall('slow', 'sleep_echo', {'text': 'first', 'delay': 0.02}), ToolCall('fast', 'sleep_echo', {'text': 'second', 'delay': 0.0})]
                yield ProviderEvent(type='done', message=Message(role='assistant', tool_calls=calls, stop_reason='tool_use'))
            else:
                yield ProviderEvent(type='done', message=Message(role='assistant', content='ok', stop_reason='stop'))

    async def run():
        completion_order = []

        async def sleep_echo(_id, args, _ctx):
            await asyncio.sleep(args['delay'])
            return ToolResult(args['text'])

        tool = AgentTool(
            'sleep_echo', 'sleep then echo',
            {'type': 'object', 'properties': {'text': {'type': 'string'}, 'delay': {'type': 'number'}}, 'required': ['text', 'delay']},
            sleep_echo,
        )
        agent = Agent(provider=Provider(), model='fake', tools=[tool], tool_execution='parallel')
        agent.subscribe(lambda e: completion_order.append(e.data['tool_call_id']) if e.type == 'tool_execution_end' else None)
        await agent.prompt('go')
        tool_messages = [m for m in agent.state.messages if m.role == 'tool_result']
        assert completion_order[:2] == ['fast', 'slow']
        assert [m.content for m in tool_messages] == ['first', 'second']

    asyncio.run(run())

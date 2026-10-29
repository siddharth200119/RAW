import pytest
from typing import List

from RAW.agent import Agent
from RAW.llms.base import BaseLLM
from RAW.models import LLMCapability, Message, Tool, ToolCall, ToolParam


class FakeLLM(BaseLLM):
    """Replays scripted assistant messages and records what each call saw."""

    def __init__(self, responses: List[Message]):
        self.responses = list(responses)
        self.calls: List[List[Message]] = []
        self.capabilities = [LLMCapability.CHAT, LLMCapability.TOOLS, LLMCapability.STREAMING]

    async def chat(self, messages, schema=None, stream=False, tools=None):
        self.calls.append([m.model_copy(deep=True) for m in messages])
        response = self.responses.pop(0) if self.responses else Message(role="assistant", content="done")
        if not stream:
            return response

        async def gen():
            if response.tool_calls:
                yield response
            elif response.content:
                yield Message(role="assistant", content=response.content)

        return gen()


def tool_call(name, arguments, id="call_1"):
    return Message(role="assistant", tool_calls=[ToolCall(id=id, name=name, arguments=arguments)])


def add_tool():
    async def add(x: int, y: int = 0) -> str:
        return str(x + y)

    return Tool(
        name="add",
        description="Add two integers",
        parameters=[
            ToolParam(name="x", type="integer", description="first", required=True),
            ToolParam(name="y", type="integer", description="second"),
        ],
        function=add,
    )


async def run(agent: Agent, text: str, stream: bool):
    return [event async for event in agent(text, stream=stream)]


def tool_messages(messages: List[Message]) -> List[Message]:
    return [m for m in messages if m.role == "tool"]


@pytest.mark.parametrize("stream", [False, True])
async def test_unknown_tool_gets_error_result_and_loop_continues(stream):
    llm = FakeLLM([
        tool_call("does_not_exist", {}, id="call_x"),
        Message(role="assistant", content="sorry"),
    ])
    agent = Agent(name="t", llm=llm, tools=[add_tool()])

    await run(agent, "hi", stream)

    assert len(llm.calls) == 2
    [result] = tool_messages(llm.calls[1])
    assert result.tool_call_id == "call_x"
    assert "tool 'does_not_exist' is not available" in result.content
    assert "Available tools: add" in result.content
    assert agent.messages[-1].content == "sorry"


@pytest.mark.parametrize("stream", [False, True])
async def test_invalid_arguments_are_returned_to_model(stream):
    llm = FakeLLM([
        tool_call("add", {"x": "five"}, id="call_bad"),
        tool_call("add", {"y": 1}, id="call_missing"),
        tool_call("add", {"x": 2, "y": 3}, id="call_ok"),
        Message(role="assistant", content="5"),
    ])
    agent = Agent(name="t", llm=llm, tools=[add_tool()])

    await run(agent, "add", stream)

    bad, missing, ok = tool_messages(agent.messages)
    assert bad.tool_call_id == "call_bad"
    assert "invalid arguments for tool 'add' at x" in bad.content
    assert "'x' is a required property" in missing.content
    assert ok.content == "5"


async def test_nested_arguments_reach_the_tool():
    received = {}

    def create(lines):
        received["lines"] = lines
        return "created"

    tool = Tool(
        name="create",
        description="create a document",
        parameters=[ToolParam(
            name="lines", type="array", description="lines", required=True,
            items={"type": "object", "properties": {"qty": {"type": "array", "items": {"type": "number"}}}},
        )],
        function=create,
    )
    lines = [{"qty": [1, 2.5]}, {"qty": []}]
    llm = FakeLLM([tool_call("create", {"lines": lines}), Message(role="assistant", content="ok")])

    await run(Agent(name="t", llm=llm, tools=[tool]), "go", stream=False)

    assert received["lines"] == lines


def three_tool_script(ids):
    return [
        tool_call("add", {"x": 1, "y": 2}, id=ids[0]),
        Message(role="assistant", tool_calls=[
            ToolCall(id=ids[1], name="add", arguments={"x": 3}),
            ToolCall(id=ids[2], name="add", arguments={"x": 4, "y": 5}),
        ]),
        Message(role="assistant", content="all done"),
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("async_hook", [False, True])
async def test_on_message_reload_is_identical(stream, async_hook):
    import json

    saved = []

    def save(message):
        saved.append(json.dumps(message.to_dict()))

    async def save_async(message):
        save(message)

    agent = Agent(
        name="t", llm=FakeLLM(three_tool_script(["a", "b", "c"])), tools=[add_tool()],
        on_message=save_async if async_hook else save,
    )
    await run(agent, "add some numbers", stream)

    # Everything after the system prompt, in loop order.
    original = [json.dumps(m.to_dict()) for m in agent.messages[1:]]
    assert saved == original
    assert [json.loads(s)["role"] for s in saved] == ["user", "assistant", "tool", "assistant", "tool", "tool", "assistant"]

    # A resumed agent sends the model exactly the same history.
    reloaded = [Message.from_dict(json.loads(s)) for s in saved]
    llm = FakeLLM([Message(role="assistant", content="again")])
    resumed = Agent(name="t", llm=llm, tools=[add_tool()], history=reloaded)
    await run(resumed, "next", stream)
    assert [json.dumps(m.to_dict()) for m in llm.calls[0][1:-1]] == original


@pytest.mark.parametrize("stream", [False, True])
async def test_tool_calls_without_ids_get_matching_ids(stream):
    agent = Agent(name="t", llm=FakeLLM(three_tool_script(["", None, ""])), tools=[add_tool()])
    await run(agent, "add", stream)

    call_ids = [tc.id for m in agent.messages if m.role == "assistant" for tc in m.tool_calls]
    result_ids = [m.tool_call_id for m in tool_messages(agent.messages)]
    assert all(call_ids) and len(set(call_ids)) == 3
    assert result_ids == call_ids


class SlowLLM(FakeLLM):
    def __init__(self, delay):
        super().__init__([])
        self.delay = delay

    async def chat(self, messages, schema=None, stream=False, tools=None):
        import asyncio
        await asyncio.sleep(self.delay)
        return await super().chat(messages, schema, stream, tools)


class LoopingLLM(FakeLLM):
    """Always asks for two more tool calls, like a confused model."""

    async def chat(self, messages, schema=None, stream=False, tools=None):
        self.responses = [Message(role="assistant", tool_calls=[
            ToolCall(id=None, name="wait", arguments={}),
            ToolCall(id=None, name="wait", arguments={}),
        ])]
        return await super().chat(messages, schema, stream, tools)


def wait_tool(seconds):
    async def wait():
        import asyncio
        await asyncio.sleep(seconds)
        return "waited"

    return Tool(name="wait", description="wait", parameters=[], function=wait)


def assert_history_valid(messages):
    open_calls = []
    for m in messages:
        if m.role == "assistant":
            assert not open_calls, "assistant message before every tool call got a result"
            open_calls = [tc.id for tc in m.tool_calls]
        elif m.role == "tool":
            assert m.tool_call_id == open_calls.pop(0)
    assert not open_calls


def time_limit_events(events):
    return [e["content"] for e in events
            if isinstance(e["content"], dict) and e["content"].get("error") == "time_limit_reached"]


def test_time_limit_defaults_to_one_hour():
    assert Agent(name="t", llm=FakeLLM([])).time_limit_seconds == 3600


@pytest.mark.parametrize("stream", [False, True])
async def test_looping_model_stops_at_time_limit(stream):
    import time

    agent = Agent(name="t", llm=LoopingLLM([]), tools=[wait_tool(0.01)], time_limit_seconds=0.2)
    start = time.monotonic()
    events = await run(agent, "loop forever", stream)

    assert time.monotonic() - start < 1
    [stop] = time_limit_events(events)
    assert events[-1]["content"] is stop
    assert stop["time_limit_seconds"] == 0.2
    assert_history_valid(agent.messages[1:])


@pytest.mark.parametrize("stream", [False, True])
async def test_hung_tool_is_cancelled_and_every_call_gets_a_result(stream):
    import time

    llm = FakeLLM([Message(role="assistant", tool_calls=[
        ToolCall(id="a", name="wait", arguments={}),
        ToolCall(id="b", name="wait", arguments={}),
    ])])
    agent = Agent(name="t", llm=llm, tools=[wait_tool(30)], time_limit_seconds=0.1)
    start = time.monotonic()
    await run(agent, "go", stream)

    assert time.monotonic() - start < 1
    results = tool_messages(agent.messages)
    assert [m.tool_call_id for m in results] == ["a", "b"]
    assert all(m.content.startswith("Cancelled") for m in results)
    assert_history_valid(agent.messages[1:])


@pytest.mark.parametrize("stream", [False, True])
async def test_hung_llm_call_stops_at_time_limit(stream):
    import time

    agent = Agent(name="t", llm=SlowLLM(30), time_limit_seconds=0.1)
    start = time.monotonic()
    events = await run(agent, "hi", stream)

    assert time.monotonic() - start < 1
    assert len(time_limit_events(events)) == 1


async def test_no_time_limit_when_disabled():
    agent = Agent(name="t", llm=SlowLLM(0.05), time_limit_seconds=None)
    events = await run(agent, "hi", stream=False)
    assert not time_limit_events(events)
    assert agent.messages[-1].content == "done"


async def test_tool_raising_its_own_timeout_is_a_tool_error():
    import asyncio

    async def flaky():
        raise asyncio.TimeoutError("upstream timed out")

    tool = Tool(name="flaky", description="d", parameters=[], function=flaky)
    llm = FakeLLM([tool_call("flaky", {}), Message(role="assistant", content="ok")])
    agent = Agent(name="t", llm=llm, tools=[tool])
    events = await run(agent, "go", stream=False)

    assert not time_limit_events(events)
    assert tool_messages(agent.messages)[0].content.startswith("Error calling tool")

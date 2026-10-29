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
    assert "Unknown tool 'does_not_exist'" in result.content
    assert "add" in result.content
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
    assert "Invalid arguments for tool 'add' at x" in bad.content
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

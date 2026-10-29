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

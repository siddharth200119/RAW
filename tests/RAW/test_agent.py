import pytest
from typing import List, Optional
from RAW.agent import Agent
from RAW.llms.base import BaseLLM
from RAW.models import LLMCapability, Message, Tool, ToolCall, Usage
from RAW.utils import Logger

# Agent's default `logger` argument is the RAW.utils.logger module, so pass a Logger.
LOGGER = Logger("test_agent")


class ScriptedLLM(BaseLLM):
    """Returns queued responses; when the queue is empty it keeps calling a tool."""

    def __init__(self, responses: Optional[List[Message]] = None, stream_responses: Optional[List[List[Message]]] = None):
        self.capabilities = [LLMCapability.COMPLETION, LLMCapability.TOOLS]
        self.responses = list(responses or [])
        self.stream_responses = list(stream_responses or [])
        self.calls = 0
        self.seen_messages: List[List[Message]] = []

    def _loop_message(self) -> Message:
        return Message(
            role="assistant",
            tool_calls=[ToolCall(id=f"call_{self.calls}", name="ping", arguments={})],
            usage=Usage(input_tokens=10, output_tokens=2)
        )

    async def chat(self, messages, schema=None, stream=False, tools=None):
        self.calls += 1
        self.seen_messages.append(list(messages))
        if stream:
            chunks = self.stream_responses.pop(0) if self.stream_responses else [self._loop_message()]

            async def gen():
                for chunk in chunks:
                    yield chunk
            return gen()
        return self.responses.pop(0) if self.responses else self._loop_message()


def _ping_tool():
    def ping():
        return "pong"
    return Tool(name="ping", description="ping", parameters=[], function=ping)


async def _run(agent, stream):
    return [chunk async for chunk in agent("go", stream=stream)]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_max_iterations_stops_tool_loop(stream):
    llm = ScriptedLLM()
    agent = Agent(name="looper", llm=llm, tools=[_ping_tool()], max_iterations=3, history=[], logger=LOGGER)

    chunks = await _run(agent, stream)

    assert llm.calls == 3
    last = chunks[-1]["content"]
    assert last["error"] == "max_iterations_reached"
    assert last["max_iterations"] == 3
    assert sum(1 for c in chunks if isinstance(c["content"], dict) and "tool_response" in c["content"]) == 3
    # Usage summed over every call in the run.
    assert agent.usage == Usage(input_tokens=30, output_tokens=6)
    assert agent.last_run_usage == agent.usage


@pytest.mark.asyncio
async def test_usage_accumulates_and_last_run_resets():
    llm = ScriptedLLM(responses=[
        Message(role="assistant", content="one", usage=Usage(input_tokens=5, output_tokens=1, cache_read_tokens=100)),
        Message(role="assistant", content="two", usage=Usage(input_tokens=7, output_tokens=3)),
    ])
    agent = Agent(name="a", llm=llm, history=[], logger=LOGGER)

    assert (await _run(agent, False))[-1]["content"] == "one"
    assert (await _run(agent, False))[-1]["content"] == "two"
    assert agent.usage == Usage(input_tokens=12, output_tokens=4, cache_read_tokens=100)
    assert agent.last_run_usage == Usage(input_tokens=7, output_tokens=3)


@pytest.mark.asyncio
async def test_unknown_tool_gets_error_result():
    llm = ScriptedLLM(responses=[
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="missing", arguments={})]),
        Message(role="assistant", content="done"),
    ])
    agent = Agent(name="a", llm=llm, history=[], logger=LOGGER)

    chunks = await _run(agent, False)

    assert chunks[-1]["content"] == "done"
    tool_messages = [m for m in llm.seen_messages[1] if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "c1"
    assert "not available" in tool_messages[0].content


@pytest.mark.asyncio
async def test_stream_keeps_provider_metadata_on_final_message():
    blocks = [{"type": "text", "text": "Hi there", "citations": [{"url": "https://example.com"}]}]
    llm = ScriptedLLM(stream_responses=[[
        Message(role="assistant", content="Hi "),
        Message(role="assistant", content="there"),
        Message(
            role="assistant",
            content=None,
            usage=Usage(input_tokens=4, output_tokens=2),
            provider_content=blocks,
            provider="anthropic",
            citations=[{"url": "https://example.com"}]
        ),
    ]])
    agent = Agent(name="a", llm=llm, history=[], logger=LOGGER)

    chunks = await _run(agent, True)

    assert "".join(c["content"]["content"] for c in chunks) == "Hi there"
    final = agent.messages[-1]
    assert final.content == "Hi there"
    assert final.provider == "anthropic"
    assert final.provider_content == blocks
    assert final.citations == [{"url": "https://example.com"}]
    assert agent.usage == Usage(input_tokens=4, output_tokens=2)

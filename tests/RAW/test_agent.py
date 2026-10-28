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


# --- Context compaction -------------------------------------------------------

from RAW.models import LLMInfo  # noqa: E402


@pytest.fixture
def no_tiktoken(monkeypatch):
    import sys
    # Importing None raises ImportError, so counts fall back to len(text) // 4.
    monkeypatch.setitem(sys.modules, "tiktoken", None)


class WindowedLLM(ScriptedLLM):
    """ScriptedLLM with a known context window that also answers summary requests."""

    def __init__(self, *args, context_window=1000, max_tokens=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = context_window
        self.max_tokens = max_tokens
        self.summary_prompts: List[str] = []

    def info(self):
        return LLMInfo(
            model_name="fake", provider="fake",
            context_window=self.window, max_tokens=self.max_tokens,
        )

    async def chat(self, messages, schema=None, stream=False, tools=None):
        if messages[0].role == "user" and messages[0].content.startswith("You are compacting"):
            self.summary_prompts.append(messages[0].content)
            return Message(role="assistant", content="SUMMARY", usage=Usage(input_tokens=3, output_tokens=1))
        return await super().chat(messages, schema=schema, stream=stream, tools=tools)


def _history(turns, tokens=100):
    out = []
    for i in range(turns):
        out.append(Message(role="user", content=f"q{i} " + "x" * tokens * 4))
        out.append(Message(role="assistant", content=f"a{i} " + "y" * tokens * 4))
    return out


def _final(content="done"):
    return Message(role="assistant", content=content, usage=Usage(input_tokens=5, output_tokens=1))


@pytest.mark.parametrize("trigger, target", [(0.5, 0.5), (0.5, 0.7), (1.0, 0.5), (0.7, 0.0)])
def test_invalid_compaction_thresholds_raise(trigger, target):
    with pytest.raises(ValueError):
        Agent(name="a", llm=ScriptedLLM(), compact_trigger=trigger, compact_target=target, logger=LOGGER)


def test_agents_do_not_share_default_lists():
    a = Agent(name="a", llm=ScriptedLLM(), logger=LOGGER)
    b = Agent(name="b", llm=ScriptedLLM(), logger=LOGGER)
    assert a.tools is not b.tools
    assert a.skills is not b.skills
    assert a.messages is not b.messages


@pytest.mark.parametrize("stream", [False, True])
async def test_compacts_at_trigger_and_uses_primary_llm_by_default(no_tiktoken, stream):
    final = [_final()]
    llm = WindowedLLM(responses=final, stream_responses=[final])
    # ~850 tokens of history against a 1000 token window: over 70%.
    agent = Agent(name="a", llm=llm, history=_history(4, 100), logger=LOGGER)

    chunks = await _run(agent, stream)

    events = [c["content"] for c in chunks if isinstance(c["content"], dict) and "compaction" in c["content"]]
    assert len(events) == 1
    info = events[0]["compaction"]
    assert info["method"] == "summarize"
    assert info["tokens_before"] > 700 and info["tokens_after"] <= 500
    assert len(llm.summary_prompts) == 1
    assert agent.context_summary == "SUMMARY"
    # The LLM saw the compacted history, and the summary sits in the system prompt.
    sent = llm.seen_messages[0]
    assert "SUMMARY" in sent[0].content
    assert len(sent) < 1 + 8 + 1
    assert sent[1].role == "user"
    assert sent[-1].content == "go"
    # Summarizer usage is counted too.
    assert agent.usage.input_tokens == 5 + 3


async def test_secondary_llm_writes_the_summary_when_given(no_tiktoken):
    primary = WindowedLLM(responses=[_final()])
    secondary = WindowedLLM(context_window=100000)
    agent = Agent(
        name="a", llm=primary, compactor_llm=secondary, history=_history(4, 100), logger=LOGGER
    )

    await _run(agent, False)

    assert len(secondary.summary_prompts) == 1
    assert primary.summary_prompts == []
    assert agent.context_summary == "SUMMARY"


async def test_no_compaction_below_trigger(no_tiktoken):
    llm = WindowedLLM(responses=[_final()], context_window=100000)
    agent = Agent(name="a", llm=llm, history=_history(4, 100), logger=LOGGER)

    chunks = await _run(agent, False)

    assert llm.summary_prompts == []
    assert agent.context_summary == ""
    assert not any(isinstance(c["content"], dict) and "compaction" in c["content"] for c in chunks)
    assert len(llm.seen_messages[0]) == 1 + 8 + 1


async def test_reply_reserve_shrinks_the_budget(no_tiktoken):
    # 850 tokens is only 42% of 2000, but max_tokens=1000 leaves a 1000 token prompt budget.
    llm = WindowedLLM(responses=[_final()], context_window=2000, max_tokens=1000)
    agent = Agent(name="a", llm=llm, history=_history(4, 100), logger=LOGGER)

    await _run(agent, False)

    assert len(llm.summary_prompts) == 1


async def test_context_window_argument_overrides_the_llm(no_tiktoken):
    llm = ScriptedLLM(responses=[_final()])  # info() raises NotImplementedError
    llm.summary_prompts = []

    async def chat(messages, schema=None, stream=False, tools=None, _orig=llm.chat):
        if messages[0].content.startswith("You are compacting"):
            llm.summary_prompts.append(messages[0].content)
            return Message(role="assistant", content="SUMMARY")
        return await _orig(messages, schema=schema, stream=stream, tools=tools)

    llm.chat = chat
    agent = Agent(name="a", llm=llm, context_window=1000, history=_history(4, 100), logger=LOGGER)

    await _run(agent, False)

    assert len(llm.summary_prompts) == 1


async def test_without_a_context_window_compaction_is_disabled(no_tiktoken):
    llm = ScriptedLLM(responses=[_final()])
    agent = Agent(name="a", llm=llm, history=_history(4, 1000), logger=LOGGER)

    await _run(agent, False)

    assert len(llm.seen_messages[0]) == 1 + 8 + 1


async def test_provider_reported_usage_triggers_compaction_mid_run(no_tiktoken):
    # The local estimate is small, but the provider says the prompt was 900 of 1000 tokens.
    llm = WindowedLLM(responses=[
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="ping", arguments={})],
            usage=Usage(input_tokens=300, cache_read_tokens=600, output_tokens=2),
        ),
        _final(),
    ], context_window=1000)
    agent = Agent(name="a", llm=llm, tools=[_ping_tool()], history=_history(3, 10), logger=LOGGER)

    chunks = await _run(agent, False)

    assert len(llm.summary_prompts) == 1
    compactions = [c for c in chunks if isinstance(c["content"], dict) and "compaction" in c["content"]]
    assert len(compactions) == 1
    # The tool call and its result survive in the second request, still paired.
    second = llm.seen_messages[1]
    tool_idx = next(i for i, m in enumerate(second) if m.role == "tool")
    assert second[tool_idx - 1].tool_calls[0].id == second[tool_idx].tool_call_id == "c1"
    assert chunks[-1]["content"] == "done"


async def test_summary_survives_the_next_call(no_tiktoken):
    llm = WindowedLLM(responses=[_final("one"), _final("two")])
    agent = Agent(name="a", llm=llm, history=_history(4, 100), logger=LOGGER)

    await _run(agent, False)
    await _run(agent, False)

    assert "SUMMARY" in agent.messages[0].content
    assert "SUMMARY" in llm.seen_messages[1][0].content

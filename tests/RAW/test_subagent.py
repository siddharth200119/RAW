import asyncio
import json
from typing import List, Optional

import pytest

from RAW.agent import Agent
from RAW.llms.base import BaseLLM
from RAW.models import LLMCapability, Message, Tool, ToolCall, Usage
from RAW.subagent import ASK_TOOL, agent_tool
from RAW.utils import InMemoryMetrics, Logger

LOGGER = Logger("test_subagent")


class ScriptLLM(BaseLLM):
    def __init__(self, responses: Optional[List[Message]] = None):
        self.capabilities = [LLMCapability.COMPLETION, LLMCapability.TOOLS]
        self.responses = list(responses or [])
        self.seen: List[List[Message]] = []
        self.tool_names: List[List[str]] = []

    async def chat(self, messages, schema=None, stream=False, tools=None):
        self.seen.append(list(messages))
        self.tool_names.append([t.name for t in tools or []])
        return self.responses.pop(0)


def call(tool, call_id="c1", /, **arguments):
    return Message(role="assistant", tool_calls=[ToolCall(id=call_id, name=tool, arguments=arguments)],
                   usage=Usage(input_tokens=10, output_tokens=1))


def final(text, input_tokens=10):
    return Message(role="assistant", content=text, usage=Usage(input_tokens=input_tokens, output_tokens=2))


def make_sub(llm, **kwargs):
    return lambda: Agent(name="sub", llm=llm, history=[], logger=LOGGER, **kwargs)


def parent_agent(llm=None, tools=None, **kwargs):
    return Agent(name="boss", llm=llm or ScriptLLM(), tools=tools or [], history=[], logger=LOGGER, **kwargs)


async def collect(tool, parent, **arguments):
    """Run a sub-agent tool the way Agent does and return everything it yielded."""
    return [item async for item in tool.function(agent=parent, **arguments)]


def tool_results(messages):
    return [m.content for m in messages if m.role == "tool"]


# --- Delegation ------------------------------------------------------------------------

async def test_parent_gets_only_the_final_answer_and_sub_context_stays_isolated():
    sub_llm = ScriptLLM([call("lookup"), final("The answer is 42")])
    lookup = Tool(name="lookup", description="l", parameters=[], function=lambda: "SECRET INTERMEDIATE DATA")
    researcher = agent_tool("researcher", "Researches things.", make_sub(sub_llm, tools=[lookup]))
    parent_llm = ScriptLLM([call("researcher", task="find the answer"), final("It is 42")])
    parent = parent_agent(parent_llm, tools=[researcher])

    chunks = [c async for c in parent("what is the answer?")]

    assert chunks[-1]["content"] == "It is 42"
    assert tool_results(parent_llm.seen[1]) == ["The answer is 42"]
    assert "SECRET INTERMEDIATE DATA" not in "".join(m.content or "" for m in parent.messages)
    # The sub-agent only saw the task, not the parent's conversation.
    assert [m.content for m in sub_llm.seen[0] if m.role == "user"] == ["find the answer"]
    assert "what is the answer?" not in "".join(m.content or "" for m in sub_llm.seen[0])


async def test_sub_agent_events_stream_to_the_caller():
    sub_llm = ScriptLLM([call("lookup"), final("done")])
    lookup = Tool(name="lookup", description="l", parameters=[], function=lambda: "x")
    researcher = agent_tool("researcher", "d", make_sub(sub_llm, tools=[lookup]))
    parent = parent_agent(ScriptLLM([call("researcher", task="t"), final("ok")]), tools=[researcher])

    chunks = [c["content"] async for c in parent("go")]

    forwarded = [c["tool_yield"] for c in chunks if isinstance(c, dict) and "tool_yield" in c]
    assert {"subagent": "researcher", "agent_ref": forwarded[0]["agent_ref"], "event": {"tool_call": "lookup", "arguments": {}}} in forwarded
    assert any(f["event"] == {"tool_response": "x"} for f in forwarded)


async def test_each_task_gets_a_fresh_sub_agent():
    sub_llm = ScriptLLM([final("one"), final("two")])
    tool = agent_tool("worker", "d", make_sub(sub_llm))
    parent = parent_agent()

    assert (await collect(tool, parent, task="first"))[-1] == "one"
    assert (await collect(tool, parent, task="second"))[-1] == "two"

    assert [m.content for m in sub_llm.seen[1] if m.role == "user"] == ["second"]


async def test_usage_rolls_up_without_double_counting_metrics():
    metrics = InMemoryMetrics()
    sub_llm = ScriptLLM([final("done", input_tokens=100)])
    tool = agent_tool("worker", "d", make_sub(sub_llm))
    parent_llm = ScriptLLM([call("worker", task="t"), final("ok")])
    parent = parent_agent(parent_llm, tools=[tool], metrics=metrics)

    [c async for c in parent("go")]

    # 10 (call) + 10 (final) from the parent plus 100 from the sub-agent.
    assert parent.usage.input_tokens == 120
    assert parent.last_run_usage.input_tokens == 120
    # The sub-agent shares the parent's metrics and reports its own tokens once, under its own name.
    assert metrics.counter("raw.llm.tokens", direction="input") == 120
    assert metrics.counter("raw.llm.tokens", agent="sub", direction="input") == 100
    assert metrics.counter("raw.subagent.calls", agent="boss", subagent="worker", status="success") == 1
    assert len(metrics.values("raw.subagent.duration_seconds")) == 1


async def test_only_the_sub_agent_gets_ask_clarification():
    sub_llm = ScriptLLM([final("done")])
    tool = agent_tool("worker", "d", make_sub(sub_llm))
    parent_llm = ScriptLLM([call("worker", task="t"), final("ok")])
    parent = parent_agent(parent_llm, tools=[tool])

    [c async for c in parent("go")]

    assert ASK_TOOL in sub_llm.tool_names[0]
    assert ASK_TOOL not in parent_llm.tool_names[0]
    assert "worker" in parent_llm.tool_names[0]


# --- Questions --------------------------------------------------------------------------

async def test_question_goes_to_the_parent_which_resumes_the_same_sub_agent():
    sub_llm = ScriptLLM([call(ASK_TOOL, question="Which year?"), final("Revenue for 2020 is 5")])
    tool = agent_tool("analyst", "d", make_sub(sub_llm))
    parent_llm = ScriptLLM([
        call("analyst", "c1", task="report revenue"),
        call("analyst", "c2", resume="PLACEHOLDER", answer="2020"),
        final("Revenue is 5"),
    ])
    parent = parent_agent(parent_llm, tools=[tool])

    # The parent LLM must quote the agent_ref it was given; patch it in once it exists.
    gen = parent("go")
    async for chunk in gen:
        content = chunk["content"]
        if isinstance(content, dict) and "tool_response" in content:
            reply = json.loads(content["tool_response"]) if content["tool_response"].startswith("{") else None
            if reply:
                assert reply["status"] == "needs_input"
                assert reply["question"] == "Which year?"
                parent_llm.responses[0].tool_calls[0].arguments["resume"] = reply["agent_ref"]
    assert parent.messages[-1].content == "Revenue is 5"

    # The sub-agent continued its own history: ask -> tool result -> the answer as a user message.
    resumed = sub_llm.seen[1]
    assert [m.role for m in resumed if m.role != "system"] == ["user", "assistant", "tool", "user"]
    assert resumed[-1].content == "2020"
    assert resumed[-3].tool_calls[0].name == ASK_TOOL
    assert tool_results(parent_llm.seen[2])[-1] == "Revenue for 2020 is 5"


async def test_on_question_answers_without_involving_the_parent():
    asked = []

    async def on_question(question, who):
        asked.append((question, who))
        return "2020"

    sub_llm = ScriptLLM([call(ASK_TOOL, question="Which year?"), final("Revenue for 2020 is 5")])
    tool = agent_tool("analyst", "d", make_sub(sub_llm), on_question=on_question)

    out = await collect(tool, parent_agent(), task="report revenue")

    assert out[-1] == "Revenue for 2020 is 5"
    assert asked == [("Which year?", "analyst")]


@pytest.mark.parametrize("handler", ["none", "timeout", "raises"])
async def test_unanswered_question_falls_back_to_the_parent(handler):
    async def returns_none(q, who):
        return None

    async def slow(q, who):
        await asyncio.sleep(5)
        return "late"

    async def raises(q, who):
        raise RuntimeError("ui crashed")

    fn = {"none": returns_none, "timeout": slow, "raises": raises}[handler]
    sub_llm = ScriptLLM([call(ASK_TOOL, question="Which year?")])
    tool = agent_tool("analyst", "d", make_sub(sub_llm), on_question=fn, question_timeout=0.05)

    out = await collect(tool, parent_agent(), task="t")

    reply = json.loads(out[-1])
    assert reply["status"] == "needs_input" and reply["question"] == "Which year?"
    assert "resume=" in reply["how_to_answer"]


async def test_questions_are_capped_per_task():
    async def on_question(q, who):
        return "an answer"

    sub_llm = ScriptLLM([
        call(ASK_TOOL, "c1", question="first?"),
        call(ASK_TOOL, "c2", question="second?"),
        final("finished with an assumption"),
    ])
    tool = agent_tool("analyst", "d", make_sub(sub_llm), on_question=on_question, max_clarifications=1)

    out = await collect(tool, parent_agent(), task="t")

    assert out[-1] == "finished with an assumption"
    assert "No more questions are allowed" in tool_results(sub_llm.seen[2])[-1]


async def test_questions_are_asked_one_at_a_time():
    active = 0
    peak = 0

    async def on_question(q, who):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return "ok"

    def sub_with(question):
        return ScriptLLM([call(ASK_TOOL, question=question), final("done")])

    llms = [sub_with("a?"), sub_with("b?")]
    factories = iter(llms)
    tool = agent_tool("w", "d", lambda: Agent(name="sub", llm=next(factories), history=[], logger=LOGGER),
                      on_question=on_question)
    parent = parent_agent()

    results = await asyncio.gather(collect(tool, parent, task="1"), collect(tool, parent, task="2"))

    assert [r[-1] for r in results] == ["done", "done"]
    assert peak == 1


async def test_resume_errors_and_open_run_limit():
    tool = agent_tool("analyst", "d", lambda: Agent(
        name="sub", llm=ScriptLLM([call(ASK_TOOL, question="q?"), final("resumed")]),
        history=[], logger=LOGGER), max_open_runs=2)
    parent = parent_agent()

    assert "no waiting sub-agent" in (await collect(tool, parent, resume="nope", answer="x"))[-1]
    assert "Error: pass `task`" in (await collect(tool, parent))[-1]

    refs = [json.loads((await collect(tool, parent, task="t"))[-1])["agent_ref"] for _ in range(3)]

    # Only the two newest runs are still waiting.
    assert "no waiting sub-agent" in (await collect(tool, parent, resume=refs[0], answer="x"))[-1]
    # A resume without an answer is rejected and keeps the run waiting.
    assert "Error: pass `answer`" in (await collect(tool, parent, resume=refs[1]))[-1]
    assert (await collect(tool, parent, resume=refs[1], answer="a"))[-1] == "resumed"
    # A finished run cannot be resumed again.
    assert "no waiting sub-agent" in (await collect(tool, parent, resume=refs[1], answer="a"))[-1]
    assert (await collect(tool, parent, resume=refs[2], answer="a"))[-1] == "resumed"


# --- Limits and failures --------------------------------------------------------------------

async def test_nesting_depth_is_limited():
    innermost = agent_tool("inner", "d", make_sub(ScriptLLM([final("never")])), max_depth=1)
    middle_llm = ScriptLLM([call("inner", task="deeper"), final("middle done")])
    middle = agent_tool("middle", "d", make_sub(middle_llm, tools=[innermost]), max_depth=3)

    out = await collect(middle, parent_agent(), task="t")

    assert out[-1] == "middle done"
    assert "not be nested deeper than 1" in tool_results(middle_llm.seen[1])[-1]


async def test_sub_agent_that_hits_max_iterations_reports_an_error():
    looping = ScriptLLM([call("ping", str(i)) for i in range(5)])
    ping = Tool(name="ping", description="p", parameters=[], function=lambda: "pong")
    tool = agent_tool("looper", "d", make_sub(looping, tools=[ping], max_iterations=2))
    metrics = InMemoryMetrics()

    out = await collect(tool, parent_agent(metrics=metrics), task="t")

    assert out[-1].startswith("Error: the sub-agent 'looper' stopped without an answer")
    assert "2 LLM calls" in out[-1]
    assert metrics.counter("raw.subagent.calls", status="error") == 1


async def test_tool_description_explains_the_resume_protocol():
    tool = agent_tool("analyst", "Analyses data.", make_sub(ScriptLLM()))
    assert tool.description.startswith("Analyses data.")
    assert "needs_input" in tool.description and "resume" in tool.description
    assert [p.name for p in tool.parameters] == ["task", "resume", "answer"]

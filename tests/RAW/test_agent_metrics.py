import sys
import types
from typing import List, Optional

import pytest

from RAW.agent import Agent
from RAW.llms.base import BaseLLM
from RAW.models import LLMCapability, LLMInfo, Message, Tool, ToolCall, Usage
from RAW.utils import InMemoryMetrics, Logger, Metrics, NullMetrics, OpenTelemetryMetrics

LOGGER = Logger("test_agent_metrics")


class FakeLLM(BaseLLM):
    model = "fake-model"

    def __init__(self, responses: Optional[List[Message]] = None, stream_responses=None, fail=False,
                 context_window: Optional[int] = None):
        self.capabilities = [LLMCapability.COMPLETION, LLMCapability.TOOLS]
        self.responses = list(responses or [])
        self.stream_responses = list(stream_responses or [])
        self.fail = fail
        self.context_window = context_window

    def info(self):
        if self.context_window is None:
            raise NotImplementedError
        return LLMInfo(model_name="fake", provider="fake", context_window=self.context_window)

    async def chat(self, messages, schema=None, stream=False, tools=None):
        if messages[0].role == "user" and messages[0].content.startswith("You are compacting"):
            return Message(role="assistant", content="SUMMARY")
        if self.fail:
            raise RuntimeError("provider down")
        if stream:
            chunks = self.stream_responses.pop(0)

            async def gen():
                for chunk in chunks:
                    yield chunk
            return gen()
        return self.responses.pop(0)


def usage(i=10, o=2):
    return Usage(input_tokens=i, output_tokens=o)


def tool_call(name="ping", call_id="c1"):
    return Message(role="assistant", tool_calls=[ToolCall(id=call_id, name=name, arguments={})], usage=usage())


def final(text="done"):
    return Message(role="assistant", content=text, usage=usage(20, 3))


def ping():
    return Tool(name="ping", description="ping", parameters=[], function=lambda: "pong")


def boom():
    def fail():
        raise RuntimeError("tool broke")
    return Tool(name="boom", description="fails", parameters=[], function=fail)


async def run(agent, stream=False, message="go"):
    return [chunk async for chunk in agent(message, stream=stream)]


def make(llm, metrics, **kwargs):
    return Agent(name="a", llm=llm, history=[], logger=LOGGER, metrics=metrics, **kwargs)


async def test_run_llm_tool_and_token_metrics():
    m = InMemoryMetrics()
    agent = make(FakeLLM([tool_call(), final()]), m, tools=[ping()])

    await run(agent)

    assert m.counter("raw.agent.runs", agent="a", status="success") == 1
    assert m.counter("raw.llm.calls", agent="a", model="fake-model", status="success") == 2
    assert m.counter("raw.tool.calls", tool="ping", status="success") == 1
    assert m.counter("raw.llm.tokens", direction="input") == 30
    assert m.counter("raw.llm.tokens", direction="output") == 5
    assert m.counter("raw.llm.tokens", direction="cache_read") == 0
    assert len(m.values("raw.llm.call_duration_seconds")) == 2
    assert len(m.values("raw.tool.duration_seconds", tool="ping")) == 1
    assert len(m.values("raw.agent.run_duration_seconds")) == 1
    assert all(v >= 0 for v in m.values("raw.llm.call_duration_seconds"))


async def test_streaming_calls_are_measured_too():
    m = InMemoryMetrics()
    stream = [[Message(role="assistant", content="hi", usage=usage())]]
    agent = make(FakeLLM(stream_responses=stream), m)

    await run(agent, stream=True)

    assert m.counter("raw.llm.calls", status="success") == 1
    assert m.counter("raw.agent.runs", status="success") == 1
    assert m.counter("raw.llm.tokens", direction="input") == 10


async def test_failing_and_unknown_tools_are_labelled():
    m = InMemoryMetrics()
    llm = FakeLLM([
        Message(role="assistant", tool_calls=[
            ToolCall(id="1", name="boom", arguments={}),
            ToolCall(id="2", name="missing", arguments={}),
        ]),
        final(),
    ])
    agent = make(llm, m, tools=[boom()])

    await run(agent)

    assert m.counter("raw.tool.calls", tool="boom", status="error") == 1
    assert m.counter("raw.tool.calls", tool="missing", status="unknown") == 1
    assert m.counter("raw.agent.runs", status="success") == 1


async def test_llm_failure_is_counted_and_still_raises():
    m = InMemoryMetrics()
    agent = make(FakeLLM(fail=True), m)

    with pytest.raises(RuntimeError, match="provider down"):
        await run(agent)

    assert m.counter("raw.llm.calls", status="error") == 1
    assert m.counter("raw.agent.runs", status="error") == 1


async def test_max_iterations_and_empty_response_statuses():
    m = InMemoryMetrics()
    looping = make(FakeLLM([tool_call(call_id=str(i)) for i in range(5)]), m, tools=[ping()], max_iterations=2)
    await run(looping)
    empty = make(FakeLLM([Message(role="assistant", content=None)]), m)
    await run(empty)

    assert m.counter("raw.agent.runs", status="max_iterations") == 1
    assert m.counter("raw.agent.runs", status="empty_response") == 1


async def test_abandoned_stream_is_counted_as_cancelled():
    m = InMemoryMetrics()
    stream = [[Message(role="assistant", content="a"), Message(role="assistant", content="b")]]
    agent = make(FakeLLM(stream_responses=stream), m)

    gen = agent("go", stream=True)
    await gen.__anext__()
    await gen.aclose()

    assert m.counter("raw.agent.runs", status="cancelled") == 1


async def test_compaction_and_context_utilization_metrics(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)  # offline: len(text) // 4
    m = InMemoryMetrics()
    history = []
    for i in range(4):
        history += [Message(role="user", content="x" * 400), Message(role="assistant", content="y" * 400)]
    agent = Agent(
        name="a", llm=FakeLLM([final()], context_window=1000), history=history, logger=LOGGER, metrics=m
    )

    await run(agent)

    assert m.counter("raw.agent.compactions", agent="a", method="summarize") == 1
    assert m.values("raw.agent.compaction_tokens_saved")[0] > 0
    assert m.last("raw.agent.context_utilization", agent="a") > 0.7


async def test_a_broken_metrics_backend_never_breaks_the_run():
    class Broken:
        def increment(self, *a, **k):
            raise RuntimeError("backend down")

        histogram = gauge = increment

    agent = make(FakeLLM([tool_call(), final()]), Broken(), tools=[ping()])

    chunks = await run(agent)

    assert chunks[-1]["content"] == "done"


def test_default_metrics_are_null_and_protocol_is_satisfied():
    agent = Agent(name="a", llm=FakeLLM(), logger=LOGGER)
    assert isinstance(agent.metrics, NullMetrics)
    assert isinstance(InMemoryMetrics(), Metrics)
    assert isinstance(OpenTelemetryMetrics.__new__(OpenTelemetryMetrics), Metrics)


# --- Backends -----------------------------------------------------------------------

def test_in_memory_metrics_filter_by_tags():
    m = InMemoryMetrics()
    m.increment("c", tags={"a": "1", "b": "x"})
    m.increment("c", 2, tags={"a": "1", "b": "y"})
    m.increment("c", 5, tags={"a": "2"})
    m.histogram("h", 1.5, tags={"a": "1"})
    m.gauge("g", 3)
    m.gauge("g", 4)

    assert m.counter("c") == 8
    assert m.counter("c", a="1") == 3
    assert m.counter("c", a="1", b="y") == 2
    assert m.counter("missing") == 0
    assert m.values("h", a="1") == [1.5] and m.values("h", a="2") == []
    assert m.last("g") == 4 and m.last("nope") is None


class FakeInstrument:
    def __init__(self, log, kind, name):
        self.log, self.kind, self.name = log, kind, name

    def add(self, value, attributes=None):
        self.log.append(("counter", self.name, value, attributes))

    def record(self, value, attributes=None):
        self.log.append(("histogram", self.name, value, attributes))

    def set(self, value, attributes=None):
        self.log.append(("gauge", self.name, value, attributes))


def test_opentelemetry_adapter_maps_to_instruments(monkeypatch):
    log: list = []
    created: list = []

    class Meter:
        def _make(self, kind):
            def create(name):
                created.append((kind, name))
                return FakeInstrument(log, kind, name)
            return create

        def __getattr__(self, attr):
            return self._make(attr.replace("create_", ""))

    fake_metrics = types.ModuleType("opentelemetry.metrics")
    fake_metrics.get_meter = lambda name: Meter()
    fake_pkg = types.ModuleType("opentelemetry")
    fake_pkg.metrics = fake_metrics
    monkeypatch.setitem(sys.modules, "opentelemetry", fake_pkg)
    monkeypatch.setitem(sys.modules, "opentelemetry.metrics", fake_metrics)

    m = OpenTelemetryMetrics()
    m.increment("raw.llm.calls", tags={"status": "success"})
    m.increment("raw.llm.calls", 2)
    m.histogram("raw.llm.call_duration_seconds", 0.5, tags={"model": "x"})
    m.gauge("raw.agent.context_utilization", 0.4)

    assert log == [
        ("counter", "raw.llm.calls", 1.0, {"status": "success"}),
        ("counter", "raw.llm.calls", 2, {}),
        ("histogram", "raw.llm.call_duration_seconds", 0.5, {"model": "x"}),
        ("gauge", "raw.agent.context_utilization", 0.4, {}),
    ]
    # One instrument per (kind, name), created once.
    assert created == [
        ("counter", "raw.llm.calls"), ("histogram", "raw.llm.call_duration_seconds"),
        ("gauge", "raw.agent.context_utilization"),
    ]


def test_opentelemetry_adapter_names_the_extra_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    with pytest.raises(ImportError, match=r"raw-agents\[otel\]"):
        OpenTelemetryMetrics()

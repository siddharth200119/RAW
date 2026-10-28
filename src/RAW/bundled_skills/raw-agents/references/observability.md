# Metrics and observability

`Agent(..., metrics=backend)` emits counters, histograms and gauges through the `Metrics` protocol (`increment`, `histogram`, `gauge`, each `(name, value, *, tags=None)`). Without it nothing is emitted. A failing backend never breaks a run. Every series is tagged with `agent` (the agent's name), so sub-agents appear under their own names.

| Metric | Type | Tags | Meaning |
|---|---|---|---|
| `raw.agent.runs` | counter | `status` | `success`, `error`, `max_iterations`, `empty_response`, `halted`, `cancelled` |
| `raw.agent.run_duration_seconds` | histogram | `status` | Wall time of a run |
| `raw.llm.calls` | counter | `model`, `status` | LLM requests; `success` / `error` |
| `raw.llm.call_duration_seconds` | histogram | `model`, `status` | Request time (streams: until the end) |
| `raw.llm.tokens` | counter | `direction` | `input`, `output`, `cache_read`, `cache_write` |
| `raw.tool.calls` | counter | `tool`, `status` | `success`, `error`, `unknown` (model called a tool that is not available) |
| `raw.tool.duration_seconds` | histogram | `tool`, `status` | Tool execution time |
| `raw.agent.context_utilization` | gauge | | Prompt tokens / usable context window |
| `raw.agent.compactions` | counter | `method` | `truncate`, `summarize`, `drop` |
| `raw.agent.compaction_tokens_saved` | histogram | | Tokens removed per compaction |
| `raw.subagent.calls` | counter | `subagent`, `status` | `success`, `needs_input`, `error` |
| `raw.subagent.duration_seconds` | histogram | `subagent`, `status` | Time in a sub-agent task |

Storage backends emit their own `raw.storage.*` metrics when given `metrics=`.

## Ready-made backends

```python
from RAW import InMemoryMetrics, OpenTelemetryMetrics

metrics = InMemoryMetrics()                       # tests, local debugging
agent = Agent(name="a", llm=llm, metrics=metrics)
...
metrics.counter("raw.tool.calls", tool="search", status="error")   # sum of series whose tags include these
metrics.values("raw.llm.call_duration_seconds")                    # histogram observations
metrics.last("raw.agent.context_utilization")                      # latest gauge value

agent = Agent(name="a", llm=llm, metrics=OpenTelemetryMetrics())   # needs raw-agents[otel]
```

`OpenTelemetryMetrics` uses the global meter provider: configure the SDK and an exporter in your application. RAW emits metrics only, not traces.

## Your own backend

Any object with the three methods works (no base class). A Prometheus sketch:

```python
class PromMetrics:
    def increment(self, name, value=1.0, *, tags=None): ...
    def histogram(self, name, value, *, tags=None): ...
    def gauge(self, name, value, *, tags=None): ...
```

## Reading what happened without a backend

- Events in the stream (`tool_call`, `tool_response`, `compaction`, `error`) show the flow of one run.
- `agent.messages` is the full history, `agent.usage` / `agent.last_run_usage` the token totals.
- The default `Logger` writes JSON lines at INFO to stdout; set `level=logging.DEBUG` for request details.

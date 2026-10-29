# Agent

```python
from RAW import Agent

agent = Agent(
    name="Support",                 # required; also the name in metrics and logs
    llm=llm,                        # required; a BaseLLM
    base_prompt="You help customers.",
    tools=[...],                    # List[Tool]
    skills=[...], skills_dir=None,  # see skills.md
    router=None,                    # see "Router" below
    history=None,                   # List[Message] to start from
    max_iterations=25,
    time_limit_seconds=3600,        # wall-clock limit per call; None disables
    on_message=None,                # fn(Message), sync or async: each message added, in order
    compactor_llm=None, context_window=None, compact_trigger=0.70, compact_target=0.50,
    skill_mode="catalog", load_skill_tools=False,
    metrics=None,                   # see observability.md
    logger=None,                    # default: a Logger named after the agent (JSON lines on stdout)
)
```

`llm` must support tools for tool use; without the TOOLS capability the agent warns and sends none (and falls back to `skill_mode="inline"`).

## Running

```python
async for chunk in agent("question", stream=False, skill=None, user_summary=""):
    ...
```

- `skill=` forces one skill for this call (inline, with only that skill's tools).
- `user_files=[File, ...]`: images go to the model when it has VISION; other file types are not parsed yet.
- Chunks and events: see `SKILL.md`. `{"error": "max_iterations_reached", ...}` means the loop guard stopped a run that kept calling tools; `{"error": "time_limit_reached", ...}` means the run passed `time_limit_seconds`. Tool calls still pending get a "Cancelled" result. Sync tools can't be interrupted, only stopped after they return.
- Tool arguments are validated against the tool's parameter schema before the tool runs; an invalid or unknown call gets an error result the model can correct.
- After a run: `agent.messages` (history), `agent.usage`, `agent.last_run_usage`, `agent.context_summary`.
- RAW stores nothing. `on_message` gets every message the run adds (user, assistant, tool), in order and with tool call ids filled in, so you can persist as you go. To continue later, save `agent.messages[1:]` (everything after the system message) and pass it as `history=` to a new agent; if you use compaction also save `agent.context_summary` and assign it back (`agent.context_summary = saved`) before the next run.

## Tools

```python
from RAW.models import Tool, ToolParam

Tool(
    name="search",
    description="Search the knowledge base.",
    parameters=[
        ToolParam(name="query", type="string", description="Search text", required=True),
        ToolParam(name="kind", type="string", description="Document kind", enums=["faq", "guide"]),
        ToolParam(name="ids", type="array", description="Ids", items={"type": "string"}),
    ],
    function=search,
)
```

`function` can be a plain function, an `async` function, a generator or an async generator. For generators every yield except the last is streamed to the caller as `{"tool_yield": value}`; the last yield is the result given to the model. Results are passed to the model as `str(result)`. Declare a parameter named `agent` on the function to receive the running `Agent` (it is not part of the schema). Arguments come from the model as keyword arguments; exceptions become an `Error calling tool: ...` result and the run continues. A call to a tool that is not available also gets an error result.

## Router: choose tools and skills per message

```python
from RAW.utils import Router

router = Router(llm=cheap_llm)                                   # LLM picks tools/skills and keeps a conversation summary
router = Router(decider=OpenRouterDecider(), decider_threshold=0.5)  # yes/no probability per tool/skill (see deciders.md)
agent = Agent(name="a", llm=llm, tools=tools, skills=skills, router=router)
```

Give exactly one of `llm` or `decider`. Without a router every tool and skill is offered. Router-selected skills are injected inline.

## Context compaction

Long runs fill the context window. Before every LLM call the agent estimates the prompt size (the provider's reported usage from the previous call plus an estimate for new messages) against `context_window - llm max_tokens`. Past `compact_trigger` (70%) it shrinks to `compact_target` (50%):

1. old bulky tool results become a one-line placeholder;
2. if still too big, the oldest turns are summarized by `compactor_llm` (or `llm` when `None`) and recent turns stay verbatim. A tool call is never separated from its result;
3. the summary goes into the system prompt and later compactions fold into it. If the summarizer fails, the oldest messages are dropped and the run continues.

```python
agent = Agent(name="r", llm=llm, compactor_llm=small_llm, context_window=200_000)
async for chunk in agent("..."):
    c = chunk["content"]
    if isinstance(c, dict) and "compaction" in c:
        print(c["compaction"])     # {"method", "tokens_before", "tokens_after", "summarized_messages"}
```

`context_window` is only needed when `llm.info().context_window` is `None` (otherwise compaction is off and a warning is logged). vLLM reads it from the server; Anthropic and Gemini have built-in values. `OpenAILLM.info()` reports a fixed 16384, which is too small for current models, so pass `context_window=` explicitly with OpenAI. Summarizer tokens are added to `agent.usage`. `compact_target < compact_trigger < 1` is enforced.

## Testing agents without a network

Subclass `BaseLLM`, implement `async chat`, set `capabilities`, and return `Message`s with `tool_calls` to script a conversation (see `providers.md`). Pass `metrics=InMemoryMetrics()` to assert on calls, tools and tokens.

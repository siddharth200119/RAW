---
name: raw-agents
description: Use when writing or debugging Python code that uses the `raw-agents` package (import name `RAW`) - LLM providers (Gemini, OpenAI, Anthropic, Groq, vLLM), the tool-calling `Agent`, skills, sub-agents, context compaction, metrics, decision models (Jev) and `Router`, or file storage. Also use when the user asks how to build an agent or call an LLM with this library.
license: MIT
---

# Building with raw-agents (`import RAW`)

`raw-agents` is an async Python library with one interface over LLM providers, an `Agent` that calls tools, Claude-format skills, sub-agents and context compaction. It needs Python 3.10+. Everything is `async`; run it with `asyncio.run(main())`.

## Install tiers - check these first

| Install | Gives you | Import fails with |
|---|---|---|
| `pip install raw-agents` | `RAW.models`, `RAW.utils` (Router, storage, metrics, token counting), `RAW.deciders` | - |
| `pip install "raw-agents[agentic]"` | + `RAW.llms`, `RAW.Agent`, `agent_tool`, images, exact token counts | `ImportError` naming the extra |
| `pip install "raw-agents[s3]"` | + `S3Storage` | |
| `pip install "raw-agents[otel]"` | + `OpenTelemetryMetrics` | |

If an import raises `ImportError` mentioning an extra, install that extra; do not work around it. Most agent code needs `[agentic]`.

## The core loop: LLM -> Tool -> Agent

```python
import asyncio
from RAW import Agent, Tool
from RAW.llms import AnthropicLLM          # or GeminiLLM, OpenAILLM, GroqLLM, VLLM
from RAW.models import ToolParam


def get_weather(city: str) -> str:         # sync, async, or (async) generator functions all work
    return f"Sunny in {city}"


weather = Tool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters=[ToolParam(name="city", type="string", description="City name", required=True)],
    function=get_weather,
)


async def main() -> None:
    llm = AnthropicLLM(model="claude-sonnet-5-5")        # reads ANTHROPIC_API_KEY
    agent = Agent(name="WeatherBot", base_prompt="Be brief.", llm=llm, tools=[weather])
    async for chunk in agent("What's the weather in Paris?"):
        content = chunk["content"]
        if isinstance(content, str):
            print(content)                                # the final answer


asyncio.run(main())
```

- `Agent(...)` is callable and is an **async generator**: `async for chunk in agent(text, stream=False)`.
- Every chunk is `{"agent_name": ..., "content": ...}`. With `stream=False` the final answer is a `str`. With `stream=True` text arrives in pieces as dicts with `"role"` and `"content"` keys (`content["content"]` is the piece).
- Other `content` dicts are events: `{"tool_call", "arguments"}`, `{"tool_response"}`, `{"tool_yield"}`, `{"compaction": {...}}`, `{"error": "max_iterations_reached", ...}`, `{"error": "time_limit_reached", ...}`. Tell them apart by key (`"role" in content` means streamed text) and check `isinstance(content, dict)` before indexing.
- `agent.usage` is the token usage over the agent's lifetime, `agent.last_run_usage` is for the last run, `agent.messages` is the history (pass `history=[...]` to resume; RAW persists nothing).
- A tool argument named `agent` receives the running `Agent`. A tool that raises is reported to the model as `Error calling tool: ...`; it does not crash the run.
- `max_iterations` (default 25) caps LLM calls per run.

## Pick the right reference

Read only the file you need:

| Task | File |
|---|---|
| Provider constructors, options, env vars, structured output, streaming, token counts | `references/providers.md` |
| `Agent` parameters, events, tools in depth, history, `Router` | `references/agent.md` |
| Writing or loading skills (`SKILL.md` folders), `skill_mode`, bundled skills | `references/skills.md` |
| Delegating to sub-agents, questions and `resume` | `references/subagents.md` |
| Long conversations: context window and compaction | `references/agent.md` (section "Context compaction") |
| Metrics, tracing what an agent did, OpenTelemetry | `references/observability.md` |
| Decision models (Jev), typed yes/no/choice/score answers, routing with a decider | `references/deciders.md` |
| `File`, `LocalStorage`, `S3Storage` | `references/storage.md` |

## Pitfalls that cost people time

- **Forgetting `await`/`async for`.** `llm.chat(...)`, `agent(...)` and `decider.decide(...)` are async. `llm.info()` is the one sync call and never touches the network.
- **`llm.chat(..., stream=True)` must be awaited first:** `async for chunk in await llm.chat(messages, stream=True)`.
- **Skills default to an on-demand catalog.** With `skill_mode="catalog"` (the default) the model sees skill names and loads one with the built-in `load_skill` tool; use `skill_mode="inline"` to put every skill in the system prompt. The names `load_skill`, `read_skill_resource` and `ask_clarification` are reserved - do not name your own tools that.
- **Skill `tools.py` only runs with `load_skill_tools=True`**, because importing it executes code. Enable it only for skills you trust.
- **Compaction needs a context window.** vLLM reports it from the server; if `llm.info().context_window` is `None` the agent logs a warning and compaction stays off. Pass `Agent(context_window=...)`.
- **Sub-agents run one after another** (the parent blocks on the tool call), and give each sub-agent a complete task: it cannot see the parent's conversation.
- **Decision models return probabilities, never text.** Thresholds differ per model; pick them from labelled data.
- **Tests must not hit the network.** Use a scripted fake `BaseLLM` (subclass it and implement `async chat`) or `InMemoryMetrics` for assertions.
- **The default logger prints JSON lines to stdout at INFO**, mixed in with your own output. Quiet it with `import logging; from RAW.utils import Logger; Agent(..., logger=Logger("app", level=logging.WARNING))`. Prefer `Logger` over `print` inside library code.

## Quick answers

- Run one agent against a real model for a script: `AnthropicLLM`/`OpenAILLM`/`GeminiLLM` read their key from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY`; `GroqLLM` needs `api_key=`; `VLLM(model=..., base_url="http://host:8000")` takes no key.
- Use a cheaper model for summaries: `Agent(llm=big, compactor_llm=small)`.
- Share a folder of Claude skills: `Agent(llm=llm, skills_dir="~/.claude/skills")` (expand `~` yourself with `os.path.expanduser`).
- Expose a specialist as a tool: `agent_tool("researcher", "what it does", factory=lambda: Agent(...))`.

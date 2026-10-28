# RAW Project

A powerful and flexible LLM interface library for Gemini and other models, designed for modern Python applications, featuring a robust Agent framework.

## Features

-   **Gemini Integration**: Seamless support for Google's Gemini models (default: `gemini-1.5-flash`).
-   **Unified Interface**: Clean, consistent API for chat, completion, and tool usage.
-   **Agent Framework**: Build autonomous agents with tool-calling capabilities.
-   **Async Support**: Fully asynchronous implementation using `httpx` and `asyncio`.
-   **Tool Calling**: Easy-to-use function calling/tool usage with schema generation.
-   **Streaming**: Native support for streaming responses.

## Installation

The base install is small and contains `RAW.models` and `RAW.utils` (logger, HTTP client, router, token counting, storage and metrics):

```bash
pip install raw-agents
```

Optional extras add the heavier dependencies:

| Extra | Adds | Needed for |
|---|---|---|
| `raw-agents[s3]` | boto3 | `RAW.utils.storage.S3Storage` |
| `raw-agents[agentic]` | numpy, opencv-python, tiktoken, boto3 | `RAW.llms`, `RAW.Agent`, `RAW.models.Image`, exact token counts |
| `raw-agents[otel]` | opentelemetry-api | `RAW.OpenTelemetryMetrics` |

```bash
pip install "raw-agents[agentic]"
# or with uv
uv add "raw-agents[agentic]"
```

Using a feature without its extra raises an `ImportError` that names the extra to install. Without `tiktoken`, `count_tokens` falls back to an estimate of 4 characters per token.

## Quick Start / Documentation

### 1. Using LLMs

The project supports LLMs via the `BaseLLM` interface. The primary implementation provided is `GeminiLLM`.

#### Initialization

To use `GeminiLLM`, you need a Google Gemini API key.

```python
import os
from RAW.llms.gemini import GeminiLLM
from RAW.utils import Logger

# Check/Set API Key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set")

# Optional: Configure Logger
logger = Logger("MyLLM")

# Initialize
# Default model: gemini-1.5-flash (configurable via model argument)
llm = GeminiLLM(api_key=api_key, logger=logger, model="gemini-1.5-flash")
```

#### Basic Chat

You can interact with the LLM using the `chat` method, which accepts a list of `Message` objects.

```python
from RAW.models import Message

messages = [
    Message(role="user", content="Hello, who are you?")
]

# Non-streaming
response = await llm.chat(messages=messages)
print(response.content)

# Streaming
async for chunk in await llm.chat(messages=messages, stream=True):
    print(chunk.content)
```

#### Anthropic (Claude)

`AnthropicLLM` talks to the Messages API over HTTP. It reads `ANTHROPIC_API_KEY` by default; `base_url` is configurable.

```python
from RAW.llms import AnthropicLLM, AnthropicOptions

llm = AnthropicLLM(
    model="claude-sonnet-5-5",
    options=AnthropicOptions(
        max_tokens=16000,             # required by the API; this is the default
        cache_system_prompt=True,     # cache tools + system prompt (e.g. a Brand Fact Sheet)
        native_tools=[                # provider tools passed through as-is
            {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}
        ],
    ),
)
```

- **Prompt caching:** `cache_system_prompt=True` puts a breakpoint on the system prompt. Set `Message(..., cache_breakpoint=True)` to cache everything up to that message, or `cache_conversation=True` for the API's automatic breakpoint. `cache_ttl="1h"` for the 1-hour cache. The API allows 4 breakpoints per request.
- **Native tools:** `native_tools` (on the options or per `chat()` call) are sent alongside RAW `Tool`s. Server-tool blocks are kept in `Message.provider_content` and re-sent unchanged on the next turn; web search citations are on `Message.citations`. Long server-tool turns (`pause_turn`) are resumed automatically.
- **Structured output:** `generate(prompt, schema=...)` uses `output_config.format`; objects get `additionalProperties: false`.
- API version `2026-10-21` is sent by default, which rejects `temperature`/`top_p`/`top_k`, so `AnthropicOptions` has no sampling fields. Use `effort` and `thinking` instead.

`OpenAILLM` and `VLLM` take `native_tools` too, and send schemas as `response_format: {"type": "json_schema", ...}`. For servers without json_schema support, set `structured_output="json_object"` on `OpenAIOptions` / `VLLMOptions`.

#### Token usage

Every provider fills `Message.usage` (`input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`) on non-streaming responses. Anthropic, OpenAI and vLLM also report it when streaming, on the last yielded message. `input_tokens` is the uncached part of the prompt.

### 2. Making Tools

Tools allow the Agent to perform actions or retrieve information. A `Tool` consists of a name, description, parameters, and a python function.

#### Define the Function

Create an async function that performs the desired action.

```python
async def get_weather(location: str):
    """Fetches weather for a given location."""
    # Your logic here (e.g., API call)
    return f"The weather in {location} is sunny."
```

#### Define the Tool Definition

Wrap the function in a `Tool` object, specifying its schema using `ToolParam`.

```python
from RAW.models import Tool
from RAW.models.tools import ToolParam

weather_tool = Tool(
    name="get_weather",
    description="Get the weather for a specific location",
    parameters=[
        ToolParam(
            name="location",
            type="string",
            description="City and State, e.g. New York, NY",
            required=True
        )
    ],
    function=get_weather
)
```

### 3. Making an Agent

The `Agent` orchestrates the interaction between the LLM, Tools, and the User. It manages the conversation history and tool execution loop.

#### Initialization

Combine the LLM and Tools into an Agent.

```python
from RAW.agent import Agent

agent = Agent(
    name="WeatherBot",
    base_prompt="You are a helpful assistant that can check the weather.",
    tools=[weather_tool], # List of Tool objects
    llm=llm,              # The LLM instance
    logger=logger         # Optional logger
)
```

#### Running the Agent

The agent is callable. You can run it in a loop to handle user input.

```python
user_input = "What is the weather in London?"

# The agent returns a generator yielding chunks of the response
async for chunk in agent(user_input, stream=True):
    # 'chunk' can be a dictionary containing content or tool execution status
    if isinstance(chunk, dict) and 'content' in chunk:
        content_obj = chunk['content']
        # content_obj might be a Message dump or a string
        if isinstance(content_obj, dict) and 'content' in content_obj:
             print(content_obj['content'], end="", flush=True)
```

#### Loop guard and usage

`max_iterations` (default 25) caps the LLM calls in one agent run. When it is reached the agent logs a warning, yields `{"error": "max_iterations_reached", "max_iterations": 25, "message": ...}` and stops.

```python
agent = Agent(name="Scheduler", llm=llm, logger=logger, max_iterations=10)

async for chunk in agent("Plan this week's posts"):
    ...

print(agent.last_run_usage)  # usage for the last call to agent(...)
print(agent.usage)           # usage summed over the agent's lifetime
```

#### Skills

A skill is a folder with a `SKILL.md`, the same format Claude uses, so skills written for Claude work here too:

```
skills/meeting-notes/
  SKILL.md           # frontmatter (name, description) + instructions in Markdown
  references/*.md    # optional files the instructions point to
  tools.py           # optional, RAW only: TOOLS = [Tool(...), ...]
```

```python
agent = Agent(name="Assistant", llm=llm, skills_dir="skills")  # a folder of skills, a skill folder, or a list
```

Loading is lenient. Only `name` and `description` matter: `name` defaults to the folder name and `description` to the first paragraph of the instructions, so a `SKILL.md` without frontmatter still loads. Other keys (`license`, `allowed-tools`, `metadata`, ...) are kept in `skill.metadata` and otherwise ignored, and RAW's own optional keys are `version`, `author`, `tags`, `triggers`, `exclusions` and `workflow`. Code-defined `Skill(...)` objects keep working, and `Skill.from_directory(path)` / `discover_skills(paths)` load skills without an agent.

How the agent uses skills (`skill_mode`):

- **`"catalog"` (default):** the system prompt lists each skill's name and description. The model calls the built-in `load_skill` tool to read a skill's full instructions, which also unlocks that skill's tools. Loaded skills stay loaded for the agent's lifetime.
- **`"inline"`:** every skill's full prompt goes into the system prompt, and all skill tools are available from the start.
- Skills picked by a `Router`, or passed as `agent(..., skill=skill)`, are always inline.

Files in the skill folder (references, templates, scripts) are listed in the loaded skill's prompt and read with the built-in `read_skill_resource` tool, which refuses anything outside the skill folder. Agents never run a skill's `scripts/`; wrap what you need as a `Tool`.

`tools.py` is Python that runs on import, so it is only imported with `Agent(..., load_skill_tools=True)` or `Skill.from_directory(path, load_tools=True)`. Do that only for skills you trust. The names `load_skill` and `read_skill_resource` are reserved. See `examples/skills_agent.py`.

#### Context compaction

Long runs fill the model's context window. The agent checks the prompt size before every LLM call, using the provider-reported usage of the previous call plus an estimate for what was added since. The usable window is `context_window` minus the reply reserve (`max_tokens`). When the prompt passes `compact_trigger` (default 70%) of it, older history is shrunk to `compact_target` (default 50%):

1. Bulky tool results from earlier turns are replaced by a one-line placeholder.
2. If that is not enough, the oldest messages are summarized and the recent turns stay verbatim. A tool call is never separated from its result.
3. The summary is shown to the model in the system prompt, and later compactions fold into it.

```python
agent = Agent(
    name="Researcher",
    llm=llm,
    compactor_llm=cheap_llm,  # writes the summaries; defaults to `llm` when None
    context_window=200_000,   # optional: only needed when llm.info() has no context window (see below)
)

async for chunk in agent("Read these 40 files and ..."):
    content = chunk["content"]
    if isinstance(content, dict) and "compaction" in content:
        print(content["compaction"])  # {"method", "tokens_before", "tokens_after", "summarized_messages"}
```

vLLM reads the window from the server's `max_model_len` (`GET /v1/models`) when the client is created; if the server doesn't report one, `info().context_window` is `None` and you can pass `context_window=` to the agent.

If the summarizer fails, the oldest messages are dropped instead, so a failure never stops the run. Summarizer usage is added to `agent.usage`. When no context window is known the agent logs a warning and compaction stays off. The history itself is not persisted by the library: read `agent.messages` and `agent.context_summary` after a run and pass `history=` to resume.

#### Sub-agents

`agent_tool` turns an agent into a tool another agent can call. The sub-agent works in its own context (its own history, tools, skills and even LLM) and the parent gets back only its final answer, so a sub-agent that reads 40 files costs the parent a few hundred tokens instead of 60k.

```python
from RAW import Agent, agent_tool

researcher = agent_tool(
    name="researcher",
    description="Finds and summarizes information in the knowledge base.",
    factory=lambda: Agent(name="researcher", llm=cheap_llm, tools=[search_tool]),  # fresh agent per task
)

boss = Agent(name="boss", llm=llm, tools=[researcher])
```

The parent waits for the sub-agent like for any other tool. If the model asks for several sub-agents in one reply they still run one after another today. Events from inside the sub-agent are forwarded to you as `{"tool_yield": {"subagent": ..., "agent_ref": ..., "event": ...}}`. The sub-agent's token usage is added to the parent's `agent.usage`, and it reports to the parent's `metrics` unless you gave it its own.

**When the sub-agent has a question.** Every sub-agent gets a built-in `ask_clarification(question)` tool. Calling it stops that sub-agent and:

1. if you passed `on_question=async (question, subagent_name) -> answer | None`, the question goes there first (a terminal prompt, a UI), one at a time, and the sub-agent carries on with the answer;
2. otherwise, or when the handler returns `None`, times out (`question_timeout`) or raises, the parent receives `{"status": "needs_input", "question": ..., "agent_ref": ...}` as the tool result. It answers from its own context, or asks the user, and then calls the tool again with `resume=<agent_ref>, answer=...`. The sub-agent continues with its history intact.

Limits: `max_clarifications` (default 2) questions per task, after which the sub-agent is told to assume and finish; `max_depth` (default 3) levels of nesting; `max_open_runs` (default 64) unanswered sub-agents kept for `resume`. Nothing is persisted: a waiting sub-agent lives as long as the tool object. See `examples/subagents.py`.

### Full Example

See `main.py` in the project root for a complete, runnable example of a Chatbot Agent.

### 4. Storage Abstraction

RAW provides a unified storage abstraction to handle files across different backends (Local, S3) without changing your application code.

#### Setup

```python
from RAW.utils.storage import LocalStorage, S3Storage
from RAW.models import File, FileType
import boto3

# Use local storage...
storage = LocalStorage(root="/tmp/my_app_uploads")

# ...or S3 storage (AWS, MinIO, Cloudflare R2, etc.)
# s3_client = boto3.client("s3", endpoint_url="http://localhost:9000", ...)
# storage = S3Storage(client=s3_client, default_bucket="my-bucket")

# Create a File object referencing the storage
my_file = File(
    name="document.pdf",
    file_type=FileType.PDF,
    storage=storage,
    location="user_uploads/document.pdf"
)
```

#### Usage

```python
# Open and read (streams automatically)
with my_file.storage.open(my_file) as stream:
    content = stream.read()

# Save (streams automatically)
# with open("local_file.txt", "rb") as f:
#     my_file.storage.save(my_file, f)

# Delete
# my_file.storage.delete(my_file)
```

### 5. Metrics Protocol

RAW components (like `Storage` or `BaseLLM`) optionally accept a `Metrics` instance. RAW defines a lightweight `Metrics` protocol so you can inject any metrics system (Prometheus, Datadog, etc.) without RAW taking on external dependencies.

#### Defining Your Metrics Backend

```python
from RAW.utils.metrics import Metrics

class MyPrometheusMetrics:
    # Example using prometheus_client or similar
    def increment(self, name: str, value: float = 1.0, *, tags: dict = None) -> None:
        print(f"Increment {name} by {value} with tags {tags}")
        
    def histogram(self, name: str, value: float, *, tags: dict = None) -> None:
        print(f"Observe {name}: {value} with tags {tags}")
        
    def gauge(self, name: str, value: float, *, tags: dict = None) -> None:
        print(f"Set gauge {name}: {value} with tags {tags}")
```

#### Injecting Metrics

Simply pass your implementation to RAW components. They will automatically emit telemetry (e.g., operation counts, duration histograms, byte transferred counters).

```python
metrics_backend = MyPrometheusMetrics()

# Now the storage backend will automatically emit metrics!
tracked_storage = LocalStorage(root="/tmp", metrics=metrics_backend)
```

#### Agent metrics

`Agent(..., metrics=backend)` reports what happens inside a run. Without it nothing is emitted, and a failing backend never breaks a run. Every series carries an `agent` tag, so sub-agents show up under their own name.

| Metric | Type | Tags | Meaning |
|---|---|---|---|
| `raw.agent.runs` | counter | `status` | Runs; `success`, `error`, `max_iterations`, `empty_response`, `halted`, `cancelled` |
| `raw.agent.run_duration_seconds` | histogram | `status` | Wall time of a run |
| `raw.llm.calls` | counter | `model`, `status` | LLM requests (`success` / `error`) |
| `raw.llm.call_duration_seconds` | histogram | `model`, `status` | Request time; for streams, until the stream ends |
| `raw.llm.tokens` | counter | `direction` | `input`, `output`, `cache_read`, `cache_write` |
| `raw.tool.calls` | counter | `tool`, `status` | `success`, `error` or `unknown` (the model called a tool that is not available) |
| `raw.tool.duration_seconds` | histogram | `tool`, `status` | Tool execution time |
| `raw.agent.context_utilization` | gauge | | Prompt tokens / usable context window before each call |
| `raw.agent.compactions` | counter | `method` | `truncate`, `summarize` or `drop` |
| `raw.agent.compaction_tokens_saved` | histogram | | Tokens removed per compaction |
| `raw.subagent.calls` | counter | `subagent`, `status` | `success`, `needs_input` or `error` |
| `raw.subagent.duration_seconds` | histogram | `subagent`, `status` | Time spent in a sub-agent task |

Two ready-made backends ship with RAW:

```python
from RAW import Agent, InMemoryMetrics, OpenTelemetryMetrics

metrics = InMemoryMetrics()                     # for tests and local debugging
agent = Agent(name="a", llm=llm, metrics=metrics)
...
metrics.counter("raw.tool.calls", tool="search", status="error")   # sums matching series
metrics.values("raw.llm.call_duration_seconds")                    # histogram observations
metrics.last("raw.agent.context_utilization")                      # latest gauge value

agent = Agent(name="a", llm=llm, metrics=OpenTelemetryMetrics())  # needs raw-agents[otel]
```

`OpenTelemetryMetrics` uses the global meter provider, so set up the SDK and an exporter in your application as usual. RAW emits metrics only, not traces.

### 6. Decision Models (Jev and other System One models)

Decision models answer typed questions about a `state` with probabilities instead of generated text. `OpenRouterDecider` calls OpenRouter's Decisions API and defaults to TypeSafe's Jev (`typesafe/jev-1.14`); any [decision model on OpenRouter](https://openrouter.ai/models?output_modalities=decisions) works by changing `model`. It needs only the base install (images in `state` need `[agentic]`).

```python
from RAW.deciders import OpenRouterDecider
from RAW.models import NoulQuestion, NoulCriteria, ChoiceQuestion, ScoreQuestion

decider = OpenRouterDecider()  # reads OPENROUTER_API_KEY

decision = await decider.decide(
    state={"ticket": "Checkout shows a blank screen after I click Pay."},
    questions={
        "is_bug": NoulQuestion(
            instructions="Is the customer reporting a software defect?",
            criteria=NoulCriteria(true="Broken product behaviour.", false="A question or feature request."),
        ),
        "team": ChoiceQuestion(
            instructions="Which team should own this ticket?",
            criteria={"payments": "Checkout or billing issues.", "frontend": "Rendering issues."},
        ),
        "urgency": ScoreQuestion(
            instructions="How urgent is this ticket?",
            criteria=["Can wait", "This week", "Blocking revenue now"],
        ),
    },
)

decision.answers["is_bug"].noul          # probability of yes, e.g. 0.96
decision.answers["team"].choice          # "payments", with .confidence and .probabilities
decision.answers["urgency"].score        # position on the scale, e.g. 1.99
decision.usage.cost                      # USD
```

`BoxQuestion` locates a described target in an image (Jev 1.14+ only). Box never reports "not found", so pair it with a `NoulQuestion`. Pick thresholds from labelled data, and re-tune them when you change model.

#### Routing with a decider

`Router` accepts either an LLM or a decider. With a decider, every tool and skill gets a yes/no question and is selected when its probability is at least `decider_threshold`:

```python
from RAW.utils import Router

router = Router(decider=OpenRouterDecider(), decider_threshold=0.5)
agent = Agent(name="support", llm=llm, tools=tools, skills=skills, router=router)
```

The result also includes `probabilities` per tool and skill for tuning the threshold. Decision models return no text, so the router passes `conversation_summary` through unchanged instead of updating it.

## License

MIT

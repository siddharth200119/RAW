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

```bash
pip install raw
```

or with `uv`:

```bash
uv add raw
```

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

### Full Example

See `main.py` in the project root for a complete, runnable example of a Chatbot Agent.

### 4. Storage Abstraction

RAW provides a unified storage abstraction to handle files across different backends (Local, S3) without changing your application code.

#### Setup

```python
from RAW.storage import LocalStorage, S3Storage
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
from RAW.metrics import Metrics

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

## License

MIT

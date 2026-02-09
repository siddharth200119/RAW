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
from RAW.modals import Message

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
from RAW.modals import Tool
from RAW.modals.tools import ToolParam

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

### Full Example

See `main.py` in the project root for a complete, runnable example of a Chatbot Agent.

## License

MIT

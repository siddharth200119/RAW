# RAW Library

A powerful and flexible LLM interface library for Gemini and other models, designed for modern Python applications.

## Features

- **Gemini Integration**: Seamless support for Google's Gemini models (including `gemini-3-flash-preview`).
- **Unified Interface**: Clean, consistent API for chat, completion, and tool usage.
- **Async Support**: Fully asynchronous implementation using `httpx` and `asyncio`.
- **Tool Calling**: Easy-to-use function calling/tool usage.
- **Streaming**: Native support for streaming responses.
- **Vision Support**: Simple API for multimodal (text + image) inputs.

## Installation

```bash
pip install raw
```

or with `uv`:

```bash
uv add raw
```

## Quick Start

### Configuration

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Basic Usage

```python
import asyncio
import os
from RAW.llms.gemini import GeminiLLM

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    llm = GeminiLLM(api_key=api_key)
    
    response = await llm.generate("Explain quantum computing in one sentence.")
    print(response)

asyncio.run(main())
```

### Chat with Tools

```python
from RAW.llms.gemini import GeminiLLM
from RAW.modals import Message, Tool, ToolParam

# Define a tool
def get_weather(city: str):
    return f"The weather in {city} is sunny."

tool = Tool(
    name="get_weather",
    description="Get weather for a city",
    parameters=[ToolParam(name="city", type="string", required=True)],
    function=get_weather
)

# Chat with tool
messages = [Message(role="user", content="What's the weather in Tokyo?")]
response = await llm.chat(messages, tools=[tool])
```

## License

MIT

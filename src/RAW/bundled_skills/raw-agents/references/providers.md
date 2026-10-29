# LLM providers

All providers subclass `BaseLLM` (`from RAW.llms import BaseLLM`) and share these methods:

| Method | Notes |
|---|---|
| `await chat(messages, schema=None, stream=False, tools=None)` | `messages` is `List[Message]`. Returns a `Message`, or with `stream=True` an async generator of `Message` chunks (await the call, then `async for`). |
| `await generate(prompt, images=None, schema=None, stream=False)` | One-shot text; with a JSON `schema` it returns a parsed `dict`. |
| `await embed(text)` | numpy array; only providers with the EMBEDDING capability. |
| `await count_tokens(text_or_messages)` | Exact where the provider supports it, else a tiktoken/length estimate. |
| `await stop()` | Close the HTTP client. |
| `info()` | **Sync, no network.** Returns `LLMInfo(model_name, provider, max_tokens, context_window, capabilities)`. |

`llm.capabilities` is a list of `LLMCapability` (`COMPLETION`, `CHAT`, `STREAMING`, `TOOLS`, `VISION`, `EMBEDDING`...). Tools and images are only sent when the capability is present.

```python
from RAW.models import Message
reply = await llm.chat([Message(role="user", content="Hello")])
print(reply.content, reply.usage)          # Usage(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
```

`Message` fields: `role` (`user`/`assistant`/`system`/`tool`), `content`, `images`, `tool_calls` (`ToolCall(id, name, arguments)`), `tool_call_id`, `usage`, `think_content`, `citations`, `cache_breakpoint`. `input_tokens` is the *uncached* part of the prompt.

## Constructors

| Class | Signature (defaults) | Key from |
|---|---|---|
| `AnthropicLLM` | `model="claude-sonnet-5-5", api_key=None, base_url="https://api.anthropic.com", options=None, logger=None` | `ANTHROPIC_API_KEY` |
| `OpenAILLM` | `model="gpt-4o-mini", api_key=None, base_url="https://api.openai.com", options=None, logger=None` | `OPENAI_API_KEY` |
| `GeminiLLM` | `api_key=None, model="gemini-2.5-flash-lite", options=None, logger=None` | `GEMINI_API_KEY` |
| `GroqLLM` | `api_key, model="llama3-8b-8192", options=None, logger=None` | pass `api_key` |
| `VLLM` | `model, base_url, options=None, logger=None` (no key) | - |

`OpenAILLM(base_url=...)` also works for any OpenAI-compatible server. `VLLM`'s constructor makes blocking probe requests (capabilities, and the context window from `GET /v1/models` `max_model_len`), so construct it once at startup rather than per request.

## Options (Pydantic models, `<Provider>Options`)

- `OpenAIOptions` / `VLLMOptions`: `temperature, top_p, max_tokens, stop, native_tools, structured_output ("json_schema" | "json_object")`.
- `GeminiOptions`: `temperature, top_p, top_k, max_output_tokens, stop_sequences`.
- `GroqOptions`: `temperature, top_p, max_tokens, stop, presence_penalty, frequency_penalty, seed`.
- `AnthropicOptions`: `max_tokens` (required by the API, default high), `stop_sequences, effort, thinking, cache_system_prompt, cache_conversation, cache_ttl ("5m"|"1h"), native_tools, fallbacks, max_pause_continuations`. No `temperature`/`top_p`/`top_k`: the default API version rejects them; use `effort` and `thinking`.

## Anthropic specifics

- Prompt caching: `cache_system_prompt=True` caches tools + system prompt; `Message(..., cache_breakpoint=True)` caches up to that message; `cache_conversation=True` uses the automatic breakpoint. The API allows 4 breakpoints per request.
- `native_tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}]` are sent as-is next to RAW `Tool`s. Server-tool blocks stay in `Message.provider_content`; citations are on `Message.citations`; `pause_turn` is resumed automatically.
- `generate(prompt, schema=...)` uses structured outputs.

## Structured output

```python
from RAW.models import jsonschema
schema = jsonschema({"title": "Out", "type": "object", "properties": {"city": {"type": "string"}}})
data = await llm.generate("Extract the city: I live in Oslo", schema=schema)   # -> dict
```

For OpenAI-compatible servers without `json_schema` support set `structured_output="json_object"`.

## Writing a fake LLM for tests

```python
from RAW.llms.base import BaseLLM
from RAW.models import LLMCapability, Message

class ScriptedLLM(BaseLLM):
    def __init__(self, replies):
        self.capabilities = [LLMCapability.COMPLETION, LLMCapability.TOOLS]
        self.replies = list(replies)
    async def chat(self, messages, schema=None, stream=False, tools=None):
        return self.replies.pop(0)          # a Message, possibly with tool_calls
```

An `Agent` only needs `capabilities` and `chat`; add `info()` returning an `LLMInfo` with `context_window` to exercise compaction.

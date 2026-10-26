import os
import codecs
import copy
import json
from pydantic import BaseModel
from typing import Any, List, Optional, Dict, Union, AsyncGenerator, Literal, Tuple
from .base import BaseLLM
from .openai import iter_sse_data
from RAW.utils import RequestsClient, Logger, count_tokens as count_tokens_util
from RAW.models import LLMCapability, Message, Image, Tool, ToolCall, LLMInfo, Usage, jsonschema


# The Messages API requires max_tokens on every request.
DEFAULT_MAX_TOKENS = 16000
ANTHROPIC_VERSION = "2026-10-21"


class AnthropicOptions(BaseModel):
    max_tokens: int = DEFAULT_MAX_TOKENS
    stop_sequences: Optional[List[str]] = None
    # output_config.effort: "low" | "medium" | "high" | "xhigh" | "max". None = model default.
    effort: Optional[str] = None
    # Raw `thinking` config, e.g. {"type": "adaptive", "display": "summarized"}.
    thinking: Optional[Dict[str, Any]] = None
    # Put a cache breakpoint on the system prompt, so tools + system are cached.
    cache_system_prompt: bool = False
    # Top-level automatic caching: the API caches up to the last cacheable block.
    cache_conversation: bool = False
    # TTL for the cache breakpoints RAW adds: None (API default, 5m) or "1h".
    cache_ttl: Optional[Literal["5m", "1h"]] = None
    # Raw Anthropic tool dicts sent alongside RAW Tools, e.g.
    # {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}.
    native_tools: List[Dict[str, Any]] = []
    # Server-side refusal fallbacks: "default" or [{"model": ...}]. Adds the beta header.
    fallbacks: Optional[Union[str, List[Dict[str, Any]]]] = None
    # How many times a `pause_turn` (long server-tool turn) is resumed automatically.
    max_pause_continuations: int = 5


_ANTHROPIC_DEFAULT_CAPABILITIES = [
    LLMCapability.COMPLETION,
    LLMCapability.CHAT,
    LLMCapability.VISION,
    LLMCapability.TOOLS,
    LLMCapability.STREAMING,
]

ANTHROPIC_MODEL_CAPABILITIES: Dict[str, List[LLMCapability]] = {
    model: list(_ANTHROPIC_DEFAULT_CAPABILITIES)
    for model in [
        "claude-fable-5-1",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-5-5",
        "claude-haiku-4-5",
    ]
}

ANTHROPIC_CONTEXT_WINDOWS: Dict[str, int] = {
    "claude-haiku-4-5": 200000,
}

# Keywords the structured-output JSON schema does not support.
_UNSUPPORTED_SCHEMA_KEYS = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "minLength", "maxLength",
}
# Keys whose values map names to sub-schemas (names must not be filtered).
_SCHEMA_MAP_KEYS = {"properties", "patternProperties", "$defs", "definitions"}


def prepare_output_schema(schema: Dict) -> Dict:
    """
    Adapt a JSON schema for Anthropic structured outputs: every object gets
    `additionalProperties: false` and unsupported constraints are dropped.
    """
    def walk(node):
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for key, value in node.items():
            if key in _UNSUPPORTED_SCHEMA_KEYS:
                continue
            if key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
                out[key] = {name: walk(sub) for name, sub in value.items()}
            else:
                out[key] = walk(value)
        if (out.get("type") == "object" or "properties" in out) and "additionalProperties" not in out:
            out["additionalProperties"] = False
        return out

    return walk(dict(schema))


class AnthropicLLM(BaseLLM):
    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com",
        options: Optional[AnthropicOptions] = None,
        logger: Optional[Logger] = None,
        anthropic_version: str = ANTHROPIC_VERSION,
        betas: Optional[List[str]] = None
    ):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        headers = {"anthropic-version": anthropic_version}
        if api_key:
            headers["x-api-key"] = api_key

        betas = list(betas or [])
        if options and options.fallbacks:
            # The scalar "default" form and the array form use different beta headers.
            fallback_beta = (
                "server-side-fallback-2026-07-01"
                if isinstance(options.fallbacks, str)
                else "server-side-fallback-2026-06-01"
            )
            if fallback_beta not in betas:
                betas.append(fallback_beta)
        if betas:
            headers["anthropic-beta"] = ",".join(betas)

        self.client = RequestsClient(
            base_url=f"{base_url.rstrip('/')}/v1",
            headers=headers,
            timeout=600,
            logger=logger
        )
        self.model = model
        self.options = options
        self.capabilities: List[LLMCapability] = list(
            ANTHROPIC_MODEL_CAPABILITIES.get(model, _ANTHROPIC_DEFAULT_CAPABILITIES)
        )
        self.logger = logger

    def _options(self) -> AnthropicOptions:
        return self.options or AnthropicOptions()

    def _cache_control(self) -> Dict:
        cache_control = {"type": "ephemeral"}
        if self._options().cache_ttl:
            cache_control["ttl"] = self._options().cache_ttl
        return cache_control

    # ------------------------------------------------------------------ #
    # Request building
    # ------------------------------------------------------------------ #

    def _mark_cache_breakpoint(self, blocks: List[Dict]) -> None:
        # Thinking blocks can't carry cache_control; mark the last other block.
        for block in reversed(blocks):
            if block.get("type") not in ("thinking", "redacted_thinking"):
                block["cache_control"] = self._cache_control()
                return

    def _convert_message_to_anthropic_format(self, message: Message) -> Tuple[str, List[Dict]]:
        """Convert a non-system RAW Message into (role, content blocks)."""
        if message.role == "tool":
            block = {"type": "tool_result", "tool_use_id": message.tool_call_id}
            if message.content:
                block["content"] = message.content
            return "user", [block]

        if message.role == "assistant":
            # Re-send our own raw blocks unchanged: server-tool results, thinking
            # signatures and citations must round-trip exactly.
            if message.provider == "anthropic" and message.provider_content:
                return "assistant", [
                    {k: v for k, v in block.items() if v is not None}
                    for block in copy.deepcopy(message.provider_content)
                ]
            blocks = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for i, tc in enumerate(message.tool_calls):
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id or f"toolu_{i}",
                    "name": tc.name,
                    "input": tc.arguments if isinstance(tc.arguments, dict) else {}
                })
            return "assistant", blocks

        # User message: images first, then the text.
        blocks = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image.to_base64()
                }
            }
            for image in message.images
        ]
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        return "user", blocks

    def _convert_messages(self, messages: List[Message]) -> Tuple[List[Dict], List[Dict]]:
        """Return (system blocks, Messages API messages)."""
        system_blocks: List[Dict] = []
        converted: List[Dict] = []

        for message in messages:
            if message.role == "system":
                if message.content:
                    block = {"type": "text", "text": message.content}
                    if message.cache_breakpoint:
                        block["cache_control"] = self._cache_control()
                    system_blocks.append(block)
                continue

            role, blocks = self._convert_message_to_anthropic_format(message)
            if not blocks:
                if self.logger:
                    self.logger.debug(f"Skipping empty {message.role} message.")
                continue
            if message.cache_breakpoint:
                self._mark_cache_breakpoint(blocks)

            # Consecutive same-role turns are merged, so parallel tool results
            # land in a single user turn (and text after them joins that turn).
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": blocks})

        if system_blocks and self._options().cache_system_prompt and "cache_control" not in system_blocks[-1]:
            system_blocks[-1]["cache_control"] = self._cache_control()

        return system_blocks, converted

    def _convert_tool_to_anthropic_format(self, tool: Tool) -> Dict:
        function = tool.to_dict()["function"]
        return {
            "name": function["name"],
            "description": function["description"],
            "input_schema": function["parameters"],
        }

    def _build_body(
        self,
        messages: List[Message],
        schema: Optional[Union[Dict, str]] = None,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        native_tools: Optional[List[Dict]] = None
    ) -> Dict:
        options = self._options()
        system_blocks, converted = self._convert_messages(messages)

        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": options.max_tokens,
            "messages": converted,
        }
        if system_blocks:
            body["system"] = system_blocks
        if stream:
            body["stream"] = True

        tool_defs = [self._convert_tool_to_anthropic_format(tool) for tool in tools or []]
        tool_defs += list(native_tools or [])
        if tool_defs:
            body["tools"] = tool_defs

        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except (ValueError, TypeError):
                schema = None
        output_config: Dict[str, Any] = {}
        if options.effort:
            output_config["effort"] = options.effort
        if isinstance(schema, dict):
            output_config["format"] = {
                "type": "json_schema",
                "schema": prepare_output_schema(schema),
            }
        if output_config:
            body["output_config"] = output_config

        if options.thinking:
            body["thinking"] = options.thinking
        if options.stop_sequences:
            body["stop_sequences"] = options.stop_sequences
        if options.cache_conversation:
            body["cache_control"] = self._cache_control()
        if options.fallbacks:
            body["fallbacks"] = options.fallbacks
        return body

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def generate(self, prompt: str, images: Optional[List[Image]] = None, schema: Optional[jsonschema] = None, stream: bool = False) -> Union[str, Dict, AsyncGenerator[Union[str, Dict], None]]:
        if not prompt:
            if self.logger:
                self.logger.warning("Prompt is empty.")

        message = Message(role="user", content=prompt, images=images or [])
        body = self._build_body([message], schema=schema, stream=stream)

        if stream:
            return self._stream_response(body)
        else:
            response = await self._get_direct_chat_response(body)
            return response.content or ""

    async def _stream_response(self, body: Dict) -> AsyncGenerator[Union[str, Dict], None]:
        async for message in self._stream_chat_response(body):
            if message.content:
                yield message.content

    async def chat(self, messages: List[Message], schema: Optional[str] = None, stream: bool = False, tools: Optional[List[Tool]] = None, native_tools: Optional[List[Dict]] = None) -> Union[Message, AsyncGenerator[Message, None]]:
        if not messages:
            if self.logger:
                self.logger.warning("Messages list is empty.")
            messages = []

        body = self._build_body(
            messages,
            schema=schema,
            stream=stream,
            tools=tools,
            native_tools=list(self._options().native_tools) + list(native_tools or []),
        )

        if stream:
            return self._stream_chat_response(body)
        else:
            return await self._get_direct_chat_response(body)

    # ------------------------------------------------------------------ #
    # Response handling
    # ------------------------------------------------------------------ #

    def _parse_content(self, blocks: List[Dict], usage: Optional[Usage]) -> Message:
        text_parts: List[str] = []
        think_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        citations: List[Dict] = []

        for block in blocks:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
                citations.extend(block.get("citations") or [])
            elif block_type == "thinking":
                if block.get("thinking"):
                    think_parts.append(block["thinking"])
            elif block_type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id"),
                    name=block["name"],
                    arguments=block.get("input") or {}
                ))
            # server_tool_use / *_tool_result blocks are kept only in provider_content.

        return Message(
            role="assistant",
            content="".join(text_parts) or None,
            think_content="\n".join(think_parts) or None,
            images=[],
            tool_calls=tool_calls,
            usage=usage,
            provider_content=blocks,
            provider="anthropic",
            citations=citations
        )

    def _check_stop_reason(self, stop_reason: Optional[str]) -> None:
        if not self.logger:
            return
        if stop_reason == "max_tokens":
            self.logger.warning(f"Anthropic response hit max_tokens ({self._options().max_tokens}); output is truncated.")
        elif stop_reason == "refusal":
            self.logger.warning("Anthropic response stopped with a refusal.")
        elif stop_reason == "pause_turn":
            self.logger.warning("Anthropic turn still paused after max_pause_continuations.")

    def _continuation_body(self, body: Dict, original_messages: List[Dict], blocks: List[Dict]) -> Dict:
        # Resume a paused turn: re-send the partial assistant turn, no extra user message.
        return {**body, "messages": original_messages + [{"role": "assistant", "content": list(blocks)}]}

    async def _post_messages(self, body: Dict) -> Dict:
        response = await self.client.post("/messages", json=body, is_async=True)
        if response.status_code >= 400:
            raise RuntimeError(f"Anthropic API error {response.status_code}: {response.text}")
        return response.json()

    async def _get_direct_chat_response(self, body: Dict) -> Message:
        try:
            original_messages = body["messages"]
            blocks: List[Dict] = []
            usage: Optional[Usage] = None
            stop_reason = None

            for _ in range(self._options().max_pause_continuations + 1):
                data = await self._post_messages(body)
                blocks.extend(data.get("content") or [])
                call_usage = Usage.from_anthropic(data.get("usage"))
                if call_usage:
                    usage = call_usage if usage is None else usage + call_usage
                stop_reason = data.get("stop_reason")
                if stop_reason != "pause_turn":
                    break
                body = self._continuation_body(body, original_messages, blocks)

            self._check_stop_reason(stop_reason)
            return self._parse_content(blocks, usage)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Chat error: {str(e)}")
            raise RuntimeError(f"Chat error: {str(e)}")

    async def _stream_events(self, body: Dict, state: Dict) -> AsyncGenerator[str, None]:
        """
        Run one streaming request. Yields text deltas; on exit `state` holds the
        assembled content blocks, usage and stop_reason.
        """
        blocks: Dict[int, Dict] = {}
        partial_json: Dict[int, str] = {}
        usage_data: Dict[str, Any] = {}
        seen_event = False
        raw = ""
        buffer = ""

        decoder = codecs.getincrementaldecoder("utf-8")()
        stream_gen = await self.client.post("/messages", json=body, stream=True, is_async=True)
        async for chunk in stream_gen:
            text = decoder.decode(chunk)
            if not seen_event and len(raw) < 65536:
                raw += text
            buffer += text
            payloads, buffer = iter_sse_data(buffer)
            for payload in payloads:
                try:
                    event = json.loads(payload)
                except ValueError:
                    continue
                seen_event = True
                event_type = event.get("type")

                if event_type == "message_start":
                    usage_data.update((event.get("message") or {}).get("usage") or {})

                elif event_type == "content_block_start":
                    index = event.get("index", 0)
                    block = dict(event.get("content_block") or {})
                    blocks[index] = block
                    if block.get("type") in ("tool_use", "server_tool_use"):
                        partial_json[index] = ""

                elif event_type == "content_block_delta":
                    index = event.get("index", 0)
                    delta = event.get("delta") or {}
                    delta_type = delta.get("type")
                    block = blocks.setdefault(index, {})
                    if delta_type == "text_delta":
                        block["text"] = block.get("text", "") + delta.get("text", "")
                        if delta.get("text"):
                            yield delta["text"]
                    elif delta_type == "input_json_delta":
                        partial_json[index] = partial_json.get(index, "") + delta.get("partial_json", "")
                    elif delta_type == "citations_delta":
                        block["citations"] = (block.get("citations") or []) + [delta.get("citation")]
                    elif delta_type == "thinking_delta":
                        block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
                    elif delta_type == "signature_delta":
                        block["signature"] = block.get("signature", "") + delta.get("signature", "")

                elif event_type == "content_block_stop":
                    index = event.get("index", 0)
                    if index in partial_json:
                        json_str = partial_json.pop(index)
                        try:
                            blocks[index]["input"] = json.loads(json_str) if json_str.strip() else {}
                        except ValueError:
                            if self.logger:
                                self.logger.warning(f"Could not parse streamed tool input: {json_str}")
                            blocks[index]["input"] = {}

                elif event_type == "message_delta":
                    stop_reason = (event.get("delta") or {}).get("stop_reason")
                    if stop_reason:
                        state["stop_reason"] = stop_reason
                    # message_delta usage counts are cumulative.
                    usage_data.update({k: v for k, v in (event.get("usage") or {}).items() if v is not None})

                elif event_type == "error":
                    raise RuntimeError(f"Anthropic stream error: {event.get('error')}")

        if not seen_event:
            # Non-SSE body, e.g. a JSON error returned for a 4xx/5xx status.
            raise RuntimeError(f"Anthropic stream returned no events: {raw[:2000]}")

        state["blocks"] = [blocks[i] for i in sorted(blocks)]
        state["usage"] = Usage.from_anthropic(usage_data)

    async def _stream_chat_response(self, body: Dict) -> AsyncGenerator[Message, None]:
        try:
            original_messages = body["messages"]
            blocks: List[Dict] = []
            usage: Optional[Usage] = None
            stop_reason = None

            for _ in range(self._options().max_pause_continuations + 1):
                state: Dict[str, Any] = {"blocks": [], "usage": None, "stop_reason": None}
                async for text in self._stream_events(body, state):
                    yield Message(role="assistant", content=text, tool_calls=[], images=[])
                blocks.extend(state["blocks"])
                if state["usage"]:
                    usage = state["usage"] if usage is None else usage + state["usage"]
                stop_reason = state["stop_reason"]
                if stop_reason != "pause_turn":
                    break
                body = self._continuation_body(body, original_messages, blocks)

            self._check_stop_reason(stop_reason)
            # Final message: tool calls, usage and raw blocks. Text was already streamed.
            final_message = self._parse_content(blocks, usage)
            final_message.content = None
            yield final_message
        except Exception as e:
            if self.logger:
                self.logger.error(f"Streaming error: {str(e)}")
            raise RuntimeError(f"Streaming error: {str(e)}")

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    def info(self) -> LLMInfo:
        return LLMInfo(
            model_name=self.model,
            provider="anthropic",
            max_tokens=self._options().max_tokens,
            context_window=ANTHROPIC_CONTEXT_WINDOWS.get(self.model, 1000000),
            capabilities=self.capabilities,
            metadata={}
        )

    async def count_tokens(self, text_or_messages: Union[str, List[Message]]) -> int:
        if isinstance(text_or_messages, str):
            messages = [Message(role="user", content=text_or_messages)]
        else:
            messages = text_or_messages
        system_blocks, converted = self._convert_messages(messages)
        body: Dict[str, Any] = {"model": self.model, "messages": converted}
        if system_blocks:
            body["system"] = system_blocks
        try:
            response = await self.client.post("/messages/count_tokens", json=body, is_async=True)
            response.raise_for_status()
            return response.json()["input_tokens"]
        except Exception:
            return count_tokens_util(text_or_messages, self.model)

    async def stop(self):
        await self.client.aclose()

    def __del__(self):
        try:
            self.client.close()
        except:
            pass

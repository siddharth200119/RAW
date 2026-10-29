import os
import codecs
import json
import re
import numpy as np
from pydantic import BaseModel
from typing import Any, List, Optional, Dict, Union, AsyncGenerator, Literal
from .base import BaseLLM
from RAW.utils import RequestsClient, Logger, count_tokens as count_tokens_util
from RAW.models import LLMCapability, Message, Image, Tool, ToolCall, LLMInfo, Usage, jsonschema


class OpenAIOptions(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    # Raw provider tool dicts appended to `tools` as-is (not executed by RAW).
    native_tools: List[Dict[str, Any]] = []
    # "json_schema" sends the schema as a real response_format; "json_object" is the
    # older JSON mode for servers that don't support json_schema.
    structured_output: Literal["json_schema", "json_object"] = "json_schema"


# OpenAIOptions / VLLMOptions fields that configure RAW and are not sent as request params.
NON_BODY_OPTIONS = {"native_tools", "structured_output"}


def build_response_format(schema: Union[Dict, str], mode: str = "json_schema") -> Dict:
    """Build an OpenAI-compatible `response_format` for a schema dict (or JSON string)."""
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except (ValueError, TypeError):
            schema = None
    if mode != "json_schema" or not isinstance(schema, dict):
        return {"type": "json_object"}
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', str(schema.get("title") or "response"))[:64]
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": dict(schema)}
    }


def iter_sse_data(buffer: str) -> tuple:
    """Split complete `data: ` lines out of an SSE text buffer. Returns (payloads, rest)."""
    *lines, rest = buffer.split("\n")
    payloads = []
    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            payloads.append(line[5:].strip())
    return payloads, rest


OPENAI_MODEL_CAPABILITIES: Dict[str, List[LLMCapability]] = {
    "gpt-4o": [LLMCapability.COMPLETION, LLMCapability.VISION, LLMCapability.TOOLS],
    "gpt-4o-mini": [LLMCapability.COMPLETION, LLMCapability.VISION, LLMCapability.TOOLS],
    "gpt-4-turbo": [LLMCapability.COMPLETION, LLMCapability.VISION, LLMCapability.TOOLS],
    "gpt-4": [LLMCapability.COMPLETION, LLMCapability.TOOLS],
    "gpt-3.5-turbo": [LLMCapability.COMPLETION, LLMCapability.TOOLS],
}


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com",
        options: Optional[OpenAIOptions] = None,
        logger: Optional[Logger] = None
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self.client = RequestsClient(
            base_url=f"{base_url}/v1",
            headers=headers,
            timeout=300,
            logger=logger
        )
        self.model = model
        self.options = options
        self.capabilities: List[LLMCapability] = OPENAI_MODEL_CAPABILITIES.get(model, [LLMCapability.COMPLETION, LLMCapability.TOOLS])
        self.logger = logger

    async def generate(self, prompt: str, images: Optional[List[Image]] = None, schema: Optional[jsonschema] = None, stream: bool = False) -> Union[str, Dict, AsyncGenerator[Union[str, Dict], None]]:
        if not prompt:
            if self.logger:
                self.logger.warning("Prompt is empty.")
        
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream
        }
        
        if self.options:
            body.update(self.options.model_dump(exclude_none=True, exclude=NON_BODY_OPTIONS))
        
        if LLMCapability.VISION in self.capabilities and images:
            body["messages"][0]["content"] = [
                {"type": "text", "text": prompt},
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image.to_base64()}"
                        },
                    }
                    for image in images
                ],
            ]
            if schema:
                for item in body["messages"][0]["content"]:
                    if item["type"] == "text":
                        item["text"] += f"\n\nGive the output in this format: {schema}"
                        break
        else:
            body["messages"][0]["content"] = [{"type": "text", "text": prompt}]
            if schema:
                body["messages"][0]["content"][0]["text"] += f"\n\nGive the output as per this json schema: {schema}"

        if schema:
            body["response_format"] = build_response_format(schema, self._options().structured_output)

        if stream:
            return self._stream_response(body)
        else:
            return await self._get_direct_response(body)

    async def _get_direct_response(self, body: Dict) -> Union[str, Dict]:
        try:
            response = await self.client.post("/chat/completions", json=body, is_async=True)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Generation error: {str(e)}")

    async def _stream_response(self, body: Dict) -> AsyncGenerator[Union[str, Dict], None]:
        try:
            stream_gen = await self.client.post("/chat/completions", json=body, stream=True, is_async=True)
            buffer = b"" 
            async for chunk in stream_gen:
                if not chunk:
                    continue
                buffer += chunk
                while b"data: " in buffer:
                    parts = buffer.split(b"data: ", 2)
                    if len(parts) < 3:
                        if buffer.count(b"data: ") == 1 and not buffer.startswith(b"data: "):
                             parts = buffer.split(b"data: ", 1)
                             buffer = b"data: " + parts[1]
                             break
                        break
                    
                    process_line = parts[1].strip()
                    if process_line == b"[DONE]":
                        return
                    
                    try:
                        data = json.loads(process_line.decode("utf-8"))
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except Exception:
                        pass
                    
                    buffer = b"data: " + parts[2]

            if buffer.startswith(b"data: "):
                final_str = buffer[len(b"data: "):].strip().decode("utf-8")
                if final_str and final_str != "[DONE]":
                    try:
                        data = json.loads(final_str)
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except Exception:
                        pass

        except Exception as e:
            raise RuntimeError(f"Stream error: {str(e)}")

    def _convert_message_to_openai_format(self, message: Message) -> Dict:
        role = message.role
        content = message.content
        
        if message.images and content:
            msg_content = [
                {"type": "text", "text": content},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image.to_base64()}"}} for image in message.images]
            ]
        else:
            msg_content = content

        msg_dict = {
            "role": role,
            "content": msg_content,
        }
        
        if role == "tool":
            msg_dict["tool_call_id"] = message.tool_call_id

        if role == "assistant" and message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": re.sub(r'[^a-zA-Z0-9_-]', '_', tc.name),
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments)
                    }
                } for i, tc in enumerate(message.tool_calls)
            ]
        return msg_dict

    def _options(self) -> OpenAIOptions:
        return self.options or OpenAIOptions()

    async def chat(self, messages: List[Message], schema: Optional[str] = None, stream: bool = False, tools: Optional[List[Tool]] = None, native_tools: Optional[List[Dict]] = None) -> Union[Message, AsyncGenerator[Message, None]]:
        if not messages:
            if self.logger:
                self.logger.warning("Messages list is empty.")
            messages = []
        
        body = {
            "model": self.model,
            "messages": [self._convert_message_to_openai_format(msg) for msg in messages],
            "stream": stream
        }
        
        if tools and LLMCapability.TOOLS in self.capabilities:
            body["tools"] = [tool.to_dict() for tool in tools]

        # Provider-native tool dicts are passed through untouched.
        extra_tools = list(self._options().native_tools) + list(native_tools or [])
        if extra_tools:
            body["tools"] = body.get("tools", []) + extra_tools

        if schema:
            body["response_format"] = build_response_format(schema, self._options().structured_output)

        if self.options:
            body.update(self.options.model_dump(exclude_none=True, exclude=NON_BODY_OPTIONS))

        if stream:
            # Ask for a final usage chunk.
            body["stream_options"] = {"include_usage": True}
            return self._stream_chat_response(body)
        else:
            return await self._get_direct_chat_response(body)
            
    async def _get_direct_chat_response(self, body: Dict) -> Message:
        try: 
            response = await self.client.post("/chat/completions", json=body, is_async=True)
            response.raise_for_status()
            data = response.json()
            message_data = data["choices"][0]["message"]
            
            tool_calls = []
            for tc in message_data.get("tool_calls", []):
                tool_calls.append(ToolCall(
                    id=tc.get("id"),
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"])
                ))
                
            return Message(
                role="assistant",
                content=message_data.get("content"),
                images=[],
                tool_calls=tool_calls,
                usage=Usage.from_openai(data.get("usage"))
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Chat error: {str(e)}")
            raise RuntimeError(f"Chat error: {str(e)}")

    async def _stream_chat_response(self, body: Dict) -> AsyncGenerator[Message, None]:
        tool_call_chunks = {}
        usage = None
        try:
            stream_gen = await self.client.post("/chat/completions", json=body, stream=True, is_async=True)
            buffer = ""
            decoder = codecs.getincrementaldecoder("utf-8")()
            async for chunk in stream_gen:
                # Buffer so an SSE line (or a UTF-8 char) split across network chunks is not dropped.
                buffer += decoder.decode(chunk)
                payloads, buffer = iter_sse_data(buffer)
                for data_str in payloads:
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                    except Exception:
                        continue

                    if data.get("usage"):
                        usage = Usage.from_openai(data["usage"])
                    
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        
                        if "content" in delta and delta["content"]:
                            yield Message(role="assistant", content=delta["content"], tool_calls=[], images=[])

                        if "tool_calls" in delta:
                            for tc_chunk in delta["tool_calls"]:
                                idx = tc_chunk.get("index", 0)
                                if idx not in tool_call_chunks:
                                    tool_call_chunks[idx] = {"id": "", "name": "", "arguments": ""}
                                
                                if "id" in tc_chunk:
                                    tool_call_chunks[idx]["id"] += tc_chunk["id"]
                                if "function" in tc_chunk:
                                    fn = tc_chunk["function"]
                                    if "name" in fn:
                                        tool_call_chunks[idx]["name"] += fn["name"]
                                    if "arguments" in fn:
                                        tool_call_chunks[idx]["arguments"] += fn["arguments"]
            
            if tool_call_chunks:
                tool_calls = []
                for idx in sorted(tool_call_chunks.keys()):
                    tc = tool_call_chunks[idx]
                    try:
                        args = json.loads(tc["arguments"])
                    except Exception:
                        args = {}
                    tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))
                
                yield Message(role="assistant", content=None, images=[], tool_calls=tool_calls, usage=usage)
            elif usage:
                # Usage-only final message (no content, no tool calls).
                yield Message(role="assistant", content=None, images=[], tool_calls=[], usage=usage)
                                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Streaming error: {str(e)}")
            raise RuntimeError(f"Streaming error: {str(e)}")

    def info(self) -> LLMInfo:
        max_tokens = None
        if self.options and self.options.max_tokens:
            max_tokens = self.options.max_tokens
        
        return LLMInfo(
            model_name=self.model,
            provider="openai",
            max_tokens=max_tokens,
            context_window=16384,
            capabilities=self.capabilities,
            metadata={}
        )

    async def count_tokens(self, text_or_messages: Union[str, List[Message]]) -> int:
        return count_tokens_util(text_or_messages, self.model)

    async def embed(self, text: str) -> np.ndarray:
        body = {"model": self.model, "input": text}
        try:
            response = await self.client.post("/embeddings", json=body, is_async=True)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if not embedding:
                raise RuntimeError("No embedding returned")
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Embedding error: {str(e)}")

    async def stop(self):
        await self.client.aclose()

    def __del__(self):
        try:
            self.client.close()
        except Exception:
            pass

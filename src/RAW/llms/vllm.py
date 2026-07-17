from .base import BaseLLM
from RAW.utils import RequestsClient, Logger, count_tokens as count_tokens_util
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, AsyncGenerator, Literal
from RAW.modals import LLMCapability, Message, Image, Tool, ToolCall, LLMInfo, jsonschema
import json
import numpy as np
import re
import httpx

class VLLMOptions(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None


_Role = Literal["user", "assistant", "system", "tool"]

class VLLM(BaseLLM):
    def __init__(
        self, 
        model: str = "Qwen/Qwen2.5-14B-Instruct-AWQ", 
        base_url: Optional[str] = None, 
        options: Optional[VLLMOptions] = None, 
        logger: Optional[Logger] = None,
        client: Optional[RequestsClient] = None
    ):
        if client is None and base_url is None:
            raise ValueError("Either 'client' or 'base_url' must be provided.")
        self.client = client or RequestsClient(
            base_url=f"{base_url}/v1",
            timeout=300,
            logger=logger
        )
        self.model = model
        self.options = options
        self.logger = logger
        self._initialize_capabilities()

    def _initialize_capabilities(self):
        try:
            # 1. Check completion support.
            response = self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1
                }
            )
            if response.status_code != 200:
                self.capabilities = [LLMCapability.COMPLETION]
                return
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Dynamic capability check failed to reach provider: {e}")
            self.capabilities = [LLMCapability.COMPLETION]
            return

        # If the completion request succeeded, check other capabilities.
        caps = [LLMCapability.COMPLETION, LLMCapability.CHAT, LLMCapability.STREAMING]

        # 2. Check tools support
        try:
            response = self.client.post("/chat/completions", json=self._tool_check_body())
            if response.status_code == 200:
                caps.append(LLMCapability.TOOLS)
        except Exception:
            pass

        # 3. Check vision support
        try:
            response = self.client.post("/chat/completions", json=self._vision_check_body())
            if response.status_code == 200:
                caps.append(LLMCapability.VISION)
        except Exception:
            pass

        # 4. Check embedding support
        try:
            response = self.client.post(
                "/embeddings",
                json={"model": self.model, "input": "ping"}
            )
            if response.status_code == 200:
                caps.append(LLMCapability.EMBEDDING)
        except Exception:
            pass

        self.capabilities = caps


    async def generate(self, prompt: str, images: Optional[List[Image]] = None, schema: Optional[jsonschema] = None, stream: bool = False) -> Union[str, Dict, AsyncGenerator[Union[str, Dict], None]]:
        if not prompt:
            self.logger.warning("Prompt is empty.")
        
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream
        }
        
        if self.options:
            body.update(self.options.model_dump(exclude_none=True))
        
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
            body["response_format"] = {
                "type": "json_object"
            }

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
                    except:
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
                    except:
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

    async def chat(self, messages: Optional[List[Message]] = None, schema: Optional[str] = None, stream: bool = False, tools: Optional[List[Tool]] = None) -> Union[Message, AsyncGenerator[Message, None]]:
        if not messages:
            self.logger.warning("Messages list is empty.")
            messages = []
        
        body = {
            "model": self.model,
            "messages": [self._convert_message_to_openai_format(msg) for msg in messages],
            "stream": stream
        }
        
        if tools and LLMCapability.TOOLS in self.capabilities:
            body["tools"] = [tool.to_dict() for tool in tools]

        if schema:
            body["response_format"] = {"type": "json_object"}

        if self.options:
            body.update(self.options.model_dump(exclude_none=True))
        
        if stream:
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
                tool_calls=tool_calls
            )
        except Exception as e:
            self.logger.error(f"Chat error: {str(e)}")
            raise RuntimeError(f"Chat error: {str(e)}")

    async def _stream_chat_response(self, body: Dict) -> AsyncGenerator[Message, None]:
        tool_call_chunks = {}
        has_tool_calls = False
                
        try:
            stream_gen = await self.client.post("/chat/completions", json=body, stream=True, is_async=True)
            async for chunk in stream_gen:
                lines = chunk.decode("utf-8").strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                    except:
                        continue
                    
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        
                        if "content" in delta and delta["content"]:
                            yield Message(role="assistant", content=delta["content"], tool_calls=[], images=[])

                        if "tool_calls" in delta:
                            has_tool_calls = True
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
                    except:
                        args = {}
                    tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))
                
                yield Message(role="assistant", content=None, images=[], tool_calls=tool_calls)
                                
        except Exception as e:
            self.logger.error(f"Streaming error: {str(e)}")
            raise RuntimeError(f"Streaming error: {str(e)}")
    
    def _tool_check_body(self) -> Dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ping_tool",
                        "description": "ping tool to check support",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {"type": "string"}
                            }
                        }
                    }
                }
            ]
        }

    def _vision_check_body(self) -> Dict:
        _PING_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        return {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "ping"},
                    {"type": "image_url", "image_url": {"url": _PING_IMAGE}}
                ]
            }],
            "max_tokens": 1
        }

    async def _check_tool_support(self) -> bool:
        try:
            response = await self.client.post("/chat/completions", json=self._tool_check_body(), is_async=True)
            if response.status_code == 200:
                return True
            if self.logger:
                self.logger.debug(f"Tool check failed with status {response.status_code}: {response.text}")
            return False
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Tool check exception: {str(e)}")
            return False

    async def _check_vision_support(self) -> bool:
        try:
            response = await self.client.post("/chat/completions", json=self._vision_check_body(), is_async=True)
            if response.status_code == 200:
                return True
            if self.logger:
                self.logger.debug(f"Vision check failed with status {response.status_code}: {response.text}")
            return False
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Vision check exception: {str(e)}")
            return False

    def info(self) -> LLMInfo:
        max_tokens = None
        if self.options and self.options.max_tokens:
            max_tokens = self.options.max_tokens
        
        return LLMInfo(
            model_name=self.model,
            provider="vllm",
            max_tokens=max_tokens,
            context_window=32768,
            capabilities=self.capabilities,
            metadata={}
        )

    async def count_tokens(self, text_or_messages: Union[str, List[Message]]) -> int:
        if isinstance(text_or_messages, str):
            prompt = text_or_messages
        else:
            prompt = "".join([m.content or "" for m in text_or_messages])
            
        body = {
            "model": self.model,
            "prompt": prompt
        }
        try:
            root_url = str(self.client.client.base_url).rstrip("/").replace("/v1", "") + "/tokenize"
            response = await self.client.post(root_url, json=body, is_async=True)
            response.raise_for_status()
            data = response.json()
            return len(data.get("tokens", []))
        except Exception:
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
        except:
            pass
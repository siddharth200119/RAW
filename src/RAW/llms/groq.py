from .base import BaseLLM
from RAW.utils import RequestsClient, Logger
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, AsyncGenerator, Literal
from RAW.modals import Message, Image, Tool, ToolCall, LLMCapability
import httpx
import json
import numpy as np
import re


# Create a default logger instance
_default_logger = Logger("GroqLLM")


class GroqOptions(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None


# Role type alias for type hints
Role = Literal["user", "assistant", "system", "tool"]


class GroqLLM(BaseLLM):
    def __init__(
        self, 
        api_key: str, 
        model: str = "llama3-8b-8192", 
        options: Optional[GroqOptions] = None,
        logger: Optional[Logger] = None
    ):
        super().__init__()
        self.logger = logger or _default_logger
        self.client = RequestsClient(
            base_url="https://api.groq.com/openai/v1",
            timeout=300,
            logger=self.logger
        )
        self.api_key = api_key
        self.model = model
        self.options = options
        self.capabilities: List[LLMCapability] = [
            LLMCapability.COMPLETION, 
            LLMCapability.TOOLS,
            LLMCapability.STREAMING
        ]

    def _build_request_body(
        self, 
        messages: List[Dict], 
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        response_format: Optional[Dict] = None
    ) -> Dict:
        """Build the request body for chat completions"""
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }
        
        if self.options:
            if self.options.temperature is not None:
                body["temperature"] = self.options.temperature
            if self.options.top_p is not None:
                body["top_p"] = self.options.top_p
            if self.options.max_tokens is not None:
                body["max_tokens"] = self.options.max_tokens
            if self.options.stop is not None:
                body["stop"] = self.options.stop
            if self.options.presence_penalty is not None:
                body["presence_penalty"] = self.options.presence_penalty
            if self.options.frequency_penalty is not None:
                body["frequency_penalty"] = self.options.frequency_penalty
            if self.options.seed is not None:
                body["seed"] = self.options.seed

        if tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice
        
        if response_format:
            body["response_format"] = response_format

        return body
    
    def _convert_message_to_groq_format(self, message: Message) -> Dict:
        """Convert Message to Groq (OpenAI) format"""
        role = message.role
        content = message.content
        
        # Handle images for vision models if needed, though Groq vision support varies.
        # Following OpenAI format: content can be list of text/image parts
        if message.images and content:
            content_parts = [{"type": "text", "text": content}]
            for image in message.images:
                image_url = f"data:{image.mime_type or 'image/jpeg'};base64,{image.to_base64()}"
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            formatted_message = {"role": role, "content": content_parts}
        else:
            formatted_message = {"role": role, "content": content}

        if role == "tool":
            formatted_message["tool_call_id"] = message.tool_call_id
            
        if role == "assistant" and message.tool_calls:
            formatted_message["tool_calls"] = [
                {
                    "id": tc.id or "call_" + str(i),  # Fallback if ID invalid, though should be present
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments)
                    }
                }
                for i, tc in enumerate(message.tool_calls)
            ]
            
        return formatted_message

    def _convert_tool_to_groq_format(self, tool: Tool) -> Dict:
        """Convert Tool to Groq (OpenAI) format"""
        # Ensure name matches regex ^[a-zA-Z0-9_-]+$ (OpenAI strictness)
        # Groq might be similar.
        name = tool.name
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
             name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.description or "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        param.name: {
                            "type": param.type,
                            "description": param.description or ""
                        }
                        for param in tool.parameters
                    },
                    "required": [
                        param.name for param in tool.parameters if param.required
                    ]
                }
            }
        }

    async def generate(
        self, 
        prompt: str, 
        images: Optional[List[Image]] = None, 
        schema: Optional[Union[str, Dict]] = None, 
        stream: bool = False
    ) -> Union[str, Dict, AsyncGenerator[Union[str, Dict], None]]:
        """Generate a response from a prompt (wrapper around chat)"""
        if not prompt:
            self.logger.warning("Prompt is empty.")

        message = Message(role="user", content=prompt, images=images or [])
        messages = [self._convert_message_to_groq_format(message)]
        
        response_format = None
        if schema:
            response_format = {"type": "json_object"}
            # detailed schema validation isn't directly supported in 'response_format' field for all models
            # but we hint json_object.
        
        body = self._build_request_body(messages, stream=stream, response_format=response_format)
        
        if stream:
            return self._stream_response(body)
        else:
            return await self._get_direct_response(body)

    async def _get_direct_response(self, body: Dict) -> Union[str, Dict]:
        """Get a direct (non-streaming) response"""
        try:
            url = "/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.client.request_async("POST", url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                return content
            else:
                raise RuntimeError("No choices in response")
                
        except httpx.HTTPStatusError as exc:
            error_text = exc.response.text if hasattr(exc.response, 'text') else str(exc)
            raise RuntimeError(f"HTTP error: {error_text}")
        except Exception as e:
            raise RuntimeError(f"Generation error: {str(e)}")

    async def _stream_response(self, body: Dict) -> AsyncGenerator[Union[str, Dict], None]:
        """Stream a response"""
        try:
            url = "/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            stream_gen = await self.client.request_async("POST", url, stream=True, json=body, headers=headers)
            buffer = ""
            async for chunk in stream_gen:
                chunk_str = chunk.decode('utf-8')
                buffer += chunk_str
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    if line.startswith('data: '):
                        line = line[6:]
                    
                    if line == '[DONE]':
                        break
                        
                    try:
                        data = json.loads(line)
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
                        
        except httpx.HTTPStatusError as exc:
            error_text = exc.response.text if hasattr(exc.response, 'text') else str(exc)
            raise RuntimeError(f"HTTP error: {error_text}")
        except Exception as e:
            raise RuntimeError(f"Stream error: {str(e)}")

    async def chat(
        self, 
        messages: List[Message], 
        schema: Optional[str] = None, 
        stream: bool = False, 
        tools: Optional[List[Tool]] = None
    ) -> Union[Message, AsyncGenerator[Message, None]]:
        """Chat with the model"""
        if tools is None:
            tools = []
            
        if not messages:
            self.logger.warning("Messages list is empty.")

        groq_messages = [self._convert_message_to_groq_format(msg) for msg in messages]
        groq_tools = [self._convert_tool_to_groq_format(tool) for tool in tools] if tools else None
        
        response_format = None
        if schema:
            response_format = {"type": "json_object"}

        body = self._build_request_body(
            groq_messages, 
            stream=stream, 
            tools=groq_tools,
            response_format=response_format
        )

        if stream:
            return self._stream_chat_response(body)
        else:
            return await self._get_direct_chat_response(body)

    async def _get_direct_chat_response(self, body: Dict) -> Message:
        """Get a direct chat response"""
        try:
            url = "/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.client.request_async("POST", url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and data["choices"]:
                message_data = data["choices"][0]["message"]
                
                content = message_data.get("content")
                tool_calls_data = message_data.get("tool_calls", [])
                
                tool_calls = []
                for tc in tool_calls_data:
                    tool_calls.append(ToolCall(
                        id=tc.get("id"),
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"])
                    ))
                
                return Message(
                    role="assistant",
                    content=content,
                    images=[],
                    tool_calls=tool_calls
                )
            else:
                raise RuntimeError("No choices in response")
                
        except httpx.HTTPStatusError as exc:
            error_text = exc.response.text if hasattr(exc.response, 'text') else str(exc)
            self.logger.error(f"HTTP error in chat response: {error_text}")
            raise RuntimeError(f"HTTP error: {error_text}")
        except Exception as e:
            self.logger.error(f"Chat error: {str(e)}")
            raise RuntimeError(f"Chat error: {str(e)}")
        
    async def _stream_chat_response(self, body: Dict) -> AsyncGenerator[Message, None]:
        """Stream a chat response"""
        try:
            url = "/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            stream_gen = await self.client.request_async("POST", url, stream=True, json=body, headers=headers)
            
            # State for accumulating tool calls across chunks
            tool_call_chunks = {}
            
            buffer = ""
            async for chunk in stream_gen:
                chunk_str = chunk.decode('utf-8')
                buffer += chunk_str
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if not line:
                        continue

                    if line.startswith('data: '):
                        line = line[6:]
                    
                    if line == '[DONE]':
                        break
                        
                    try:
                        data = json.loads(line)
                        if "error" in data:
                            error_msg = json.dumps(data["error"])
                            self.logger.error(f"Groq API Error: {error_msg}")
                            raise RuntimeError(f"Groq API Error: {error_msg}")

                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            
                            # Handle content
                            if "content" in delta and delta["content"]:
                                yield Message(
                                    role="assistant",
                                    content=delta["content"],
                                    images=[],
                                    tool_calls=[]
                                )
                            
                            # Handle tool calls (which come in chunks)
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

                    except json.JSONDecodeError:
                         continue
            
            # Yield accumulated tool calls at the end of the stream
            # Note: Streaming method usually yields partial content. 
            # Tool calls are typically atomic when executed, but streaming them structure-wise is complex.
            # Here we follow a pattern where we might yield a final message with tool calls?
            # Or we should have yielded them as they complete?
            # A common pattern in streaming agents is to yield a specific object for tool calls.
            # For strict `AsyncGenerator[Message, None]`, we can yield a message with the tool call when it's fully formed.
            
            if tool_call_chunks:
                tool_calls = []
                for idx in sorted(tool_call_chunks.keys()):
                    tc = tool_call_chunks[idx]
                    try:
                        args = json.loads(tc["arguments"])
                    except json.JSONDecodeError:
                        args = {} # Best effort or raw string?
                        
                    tool_calls.append(ToolCall(
                        id=tc["id"],
                        name=tc["name"],
                        arguments=args
                    ))
                
                yield Message(
                    role="assistant",
                    content=None,
                    images=[],
                    tool_calls=tool_calls
                )
                    
        except httpx.HTTPStatusError as exc:
            error_text = exc.response.text if hasattr(exc.response, 'text') else str(exc)
            self.logger.error(f"HTTP error in stream chat response: {error_text}")
            raise RuntimeError(f"HTTP error: {error_text}")
        except Exception as e:
            self.logger.error(f"Stream chat error: {str(e)}")
            raise RuntimeError(f"Stream error: {str(e)}")

    async def stop(self):
        """Stop and close the HTTP client"""
        await self.client.aclose()

    def __del__(self):
        """Destructor to close the client"""
        try:
            self.client.close()
        except Exception:
            pass  # Ignore errors during cleanup

    async def embed(self, text: str) -> np.ndarray:
        """Embedding is not supported by GroqLLM in this implementation"""
        raise NotImplementedError("GroqLLM does not support embedding.")

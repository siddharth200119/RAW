import os
from .base import BaseLLM
from RAW.utils import RequestsClient, Logger, count_tokens as count_tokens_util
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, AsyncGenerator, Literal
from RAW.models import Message, Image, Tool, ToolCall, LLMCapability, LLMInfo, jsonschema
import httpx
import json
import numpy as np
import re


# Create a default logger instance
_default_logger = Logger("GeminiLLM")


class GeminiOptions(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None


# Role type alias for type hints
Role = Literal["user", "assistant", "system", "tool"]


class GeminiLLM(BaseLLM):
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gemini-2.5-flash-lite", 
        options: Optional[GeminiOptions] = None,
        logger: Optional[Logger] = None
    ):
        super().__init__()
        self.logger = logger or _default_logger
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
        self.client = RequestsClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            timeout=300,
            logger=self.logger
        )
        self.api_key = api_key
        self.model = model
        self.options = options
        self.capabilities: List[LLMCapability] = [
            LLMCapability.COMPLETION, 
            LLMCapability.TOOLS,
            LLMCapability.VISION,
            LLMCapability.STREAMING
        ]

    def _build_generation_config(self) -> Dict:
        """Build generation config from options"""
        config = {}
        if self.options:
            if self.options.temperature is not None:
                config["temperature"] = self.options.temperature
            if self.options.top_p is not None:
                config["topP"] = self.options.top_p
            if self.options.top_k is not None:
                config["topK"] = self.options.top_k
            if self.options.max_output_tokens is not None:
                config["maxOutputTokens"] = self.options.max_output_tokens
            if self.options.stop_sequences:
                config["stopSequences"] = self.options.stop_sequences
        return config

    async def generate(
        self, 
        prompt: str, 
        images: Optional[List[Image]] = None, 
        schema: Optional[jsonschema] = None, 
        stream: bool = False
    ) -> Union[str, Dict, AsyncGenerator[Union[str, Dict], None]]:
        """Generate a response from a prompt"""
        if not prompt:
            self.logger.warning("Prompt is empty.")

        parts = [{"text": prompt}]
        
        if images:
            for image in images:
                parts.append({
                    "inline_data": {
                        "mime_type": image.mime_type or "image/jpeg",
                        "data": image.to_base64().split(',')[-1]
                    }
                })

        body = {
            "contents": [{"parts": parts}]
        }

        generation_config = self._build_generation_config()
        if generation_config:
            body["generationConfig"] = generation_config

        if schema:
            if not body.get("generationConfig"):
                body["generationConfig"] = {}
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = dict(schema)

        if stream:
            return self._stream_response(body)
        else:
            return await self._get_direct_response(body)

    async def _get_direct_response(self, body: Dict) -> Union[str, Dict]:
        """Get a direct (non-streaming) response"""
        try:
            url = f"/models/{self.model}:generateContent?key={self.api_key}"
            
            response = await self.client.post(url, json=body, is_async=True)
            response.raise_for_status()
            data = response.json()
            
            if "candidates" in data and data["candidates"]:
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                return content
            else:
                raise RuntimeError("No candidates in response")
                
        except httpx.HTTPStatusError as exc:
            error_text = exc.response.text if hasattr(exc.response, 'text') else str(exc)
            raise RuntimeError(f"HTTP error: {error_text}")
        except Exception as e:
            raise RuntimeError(f"Generation error: {str(e)}")

    async def _stream_response(self, body: Dict) -> AsyncGenerator[Union[str, Dict], None]:
        """Stream a response"""
        try:
            url = f"/models/{self.model}:streamGenerateContent?key={self.api_key}&alt=sse"
            
            stream_gen = await self.client.post(url, stream=True, json=body, is_async=True)
            async for chunk in stream_gen:
                chunk_str = chunk.decode('utf-8').strip()
                if not chunk_str:
                    continue
          
                if chunk_str.startswith('data: '):
                    chunk_str = chunk_str[6:]
                if chunk_str == '[DONE]':
                    break
                    
                try:
                    data = json.loads(chunk_str)
                    if "candidates" in data and data["candidates"]:
                        if "content" in data["candidates"][0]:
                            parts = data["candidates"][0]["content"]["parts"]
                            if parts and "text" in parts[0]:
                                yield parts[0]["text"]
                except json.JSONDecodeError:
                    continue
                    
        except httpx.HTTPStatusError as exc:
            error_text = exc.response.text if hasattr(exc.response, 'text') else str(exc)
            raise RuntimeError(f"HTTP error: {error_text}")
        except Exception as e:
            raise RuntimeError(f"Stream error: {str(e)}")

    def _convert_message_to_gemini_format(
        self, 
        message: Message, 
        prev_messages: Optional[List[Message]] = None
    ) -> Dict:
        """Convert Message to Gemini format"""
        role_mapping = {
            "user": "user",
            "assistant": "model",
            "system": "user",
            "tool": "function"
        }
        
        parts = []
        if message.content:
            parts.append({"text": message.content})
            
        if message.images:
            for image in message.images:
                parts.append({
                    "inline_data": {
                        "mime_type": image.mime_type or "image/jpeg",
                        "data": image.to_base64().split(',')[-1]
                    }
                })
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                parts.append({
                    "functionCall": {
                        "name": tool_call.name,
                        "args": tool_call.arguments
                    }
                })
        
        if message.role == "tool":
            # For role 'function', return a function response
            function_name = None
            if prev_messages:
                for prev_msg in reversed(prev_messages):
                    if prev_msg.role == "assistant" and prev_msg.tool_calls:
                        function_name = prev_msg.tool_calls[-1].name
                        break
            if not function_name:
                function_name = "unknown_function"
            parts = [{
                "functionResponse": {
                    "name": function_name,
                    "response": {"content": message.content}
                }
            }]
        
        return {
            "role": role_mapping.get(message.role, "user"),
            "parts": parts
        }
    
    def _convert_tool_to_gemini_format(self, tool: Tool) -> Dict:
        """Convert Tool to Gemini format with name sanitization"""
        try:
            name = tool.name
            # Sanitize function name to meet Gemini API requirements
            if not name or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_\.\-]*$', name) or len(name) > 64:
                name = f"_{name}" if name else "tool_function"
                name = re.sub(r'[^a-zA-Z0-9_\.\-]', '', name)[:64]
            return {
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
        except Exception as e:
            raise ValueError(f"Tool conversion error for {tool.name}: {str(e)}")

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

        contents = []
        for i, msg in enumerate(messages):
            prev_msgs = messages[:i] if i > 0 else None
            contents.append(self._convert_message_to_gemini_format(msg, prev_msgs))

        body = {
            "contents": contents
        }

        generation_config = self._build_generation_config()
        if generation_config:
            body["generationConfig"] = generation_config

        if tools and LLMCapability.TOOLS in self.capabilities:
            body["tools"] = [{
                "functionDeclarations": [self._convert_tool_to_gemini_format(tool) for tool in tools]
            }]

        if schema:
            if not body.get("generationConfig"):
                body["generationConfig"] = {}
            body["generationConfig"]["responseMimeType"] = "application/json"
            if isinstance(schema, dict):
                body["generationConfig"]["responseSchema"] = schema

        if stream:
            return self._stream_chat_response(body)
        else:
            return await self._get_direct_chat_response(body)

    async def _get_direct_chat_response(self, body: Dict) -> Message:
        """Get a direct chat response"""
        try:
            url = f"/models/{self.model}:generateContent?key={self.api_key}"
            
            response = await self.client.post(url, json=body, is_async=True)
            response.raise_for_status()
            data = response.json()
            
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                content_data = candidate["content"]
                
                content = ""
                tool_calls = []
                
                for part in content_data["parts"]:
                    if "text" in part:
                        content += part["text"]
                    elif "functionCall" in part:
                        func_call = part["functionCall"]
                        tool_call = ToolCall(
                            name=func_call["name"],
                            arguments=func_call.get("args", {})
                        )
                        tool_calls.append(tool_call)
                
                return Message(
                    role="assistant",
                    content=content,
                    images=[],
                    tool_calls=tool_calls
                )
            else:
                raise RuntimeError("No candidates in response")
                
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
            url = f"/models/{self.model}:streamGenerateContent?key={self.api_key}&alt=sse"
            
            stream_gen = await self.client.post(url, stream=True, json=body, is_async=True)
            async for chunk in stream_gen:
                chunk_str = chunk.decode('utf-8').strip()
                if not chunk_str:
                    continue
                    
                if chunk_str.startswith('data: '):
                    chunk_str = chunk_str[6:]
                
                if chunk_str == '[DONE]':
                    break
                    
                try:
                    data = json.loads(chunk_str)
                    if "candidates" in data and data["candidates"]:
                        candidate = data["candidates"][0]
                        if "content" in candidate:
                            content_data = candidate["content"]
                            
                            content = ""
                            tool_calls = []
                            
                            for part in content_data["parts"]:
                                if "text" in part:
                                    content += part["text"]
                                elif "functionCall" in part:
                                    func_call = part["functionCall"]
                                    tool_call = ToolCall(
                                        name=func_call["name"],
                                        arguments=func_call.get("args", {})
                                    )
                                    tool_calls.append(tool_call)
                            
                            message = Message(
                                role="assistant",
                                content=content,
                                images=[],
                                tool_calls=tool_calls
                            )
                            yield message
                            
                except json.JSONDecodeError:
                    continue
                    
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
        """Embedding is not supported by GeminiLLM"""
        raise NotImplementedError("GeminiLLM does not support embedding.")

    async def info(self) -> LLMInfo:
        """Get information about the LLM."""
        max_tokens = self.options.max_output_tokens if self.options else None
        return LLMInfo(
            model_name=self.model,
            provider="gemini",
            max_tokens=max_tokens,
            context_window=1048576,
            capabilities=self.capabilities,
            metadata={}
        )

    async def count_tokens(self, text_or_messages: Union[str, List[Message]]) -> int:
        """Count tokens in text or messages."""
        if isinstance(text_or_messages, str):
            contents = [{"parts": [{"text": text_or_messages}]}]
        else:
            contents = []
            for i, msg in enumerate(text_or_messages):
                prev_msgs = text_or_messages[:i] if i > 0 else None
                contents.append(self._convert_message_to_gemini_format(msg, prev_msgs))
                
        body = {"contents": contents}
        try:
            url = f"/models/{self.model}:countTokens?key={self.api_key}"
            response = await self.client.post(url, json=body, is_async=True)
            response.raise_for_status()
            data = response.json()
            return data.get("totalTokens", 0)
        except Exception:
            return count_tokens_util(text_or_messages, self.model)
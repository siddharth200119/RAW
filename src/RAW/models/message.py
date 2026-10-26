from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Literal
from .image import Image
from .usage import Usage


class ToolCall(BaseModel):
    id: Optional[str] = None
    name: str
    # Any JSON value: tool arguments can contain nested objects and arrays.
    arguments: Dict[str, Any]


class Message(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str] = None
    think_content: Optional[str] = None
    images: List[Image] = []
    tool_calls: List[ToolCall] = []
    tool_call_id: Optional[str] = None
    # Token usage reported by the provider for the call that produced this message.
    usage: Optional[Usage] = None
    # Raw provider content blocks (e.g. Anthropic server-tool / thinking blocks).
    # Only the provider named in `provider` re-sends them; others use the fields above.
    provider_content: Optional[List[Dict[str, Any]]] = None
    provider: Optional[str] = None
    # Citations attached to the response text (e.g. Anthropic web search results).
    citations: List[Dict[str, Any]] = []
    # Ask providers that support prompt caching to cache the prompt up to this message.
    cache_breakpoint: bool = False

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "think_content": self.think_content,
            "images": [image.to_base64() for image in self.images],
            "tool_calls": [tool_call.model_dump() for tool_call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "usage": self.usage.model_dump() if self.usage else None,
            "provider_content": self.provider_content,
            "provider": self.provider,
            "citations": self.citations,
            "cache_breakpoint": self.cache_breakpoint,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        return cls(
            role=data["role"],
            content=data.get("content"),
            think_content=data.get("think_content"),
            images=[Image.from_base64(img) for img in data.get("images", [])],
            tool_calls=[
                ToolCall(**tool_call)
                for tool_call in data.get("tool_calls", [])
            ],
            tool_call_id=data.get("tool_call_id"),
            usage=Usage(**data["usage"]) if data.get("usage") else None,
            provider_content=data.get("provider_content"),
            provider=data.get("provider"),
            citations=data.get("citations") or [],
            cache_breakpoint=data.get("cache_breakpoint", False),
        )

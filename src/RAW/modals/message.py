from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Literal
from .image import Image


class ToolCall(BaseModel):
    id: Optional[str] = None
    name: str
    arguments: Dict[
        str,
        Union[str, int, float, bool, None, List[Dict], List[str]]
    ]


class Message(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str] = None
    think_content: Optional[str] = None
    images: List[Image] = []
    tool_calls: List[ToolCall] = []
    tool_call_id: Optional[str] = None

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "think_content": self.think_content,
            "images": [image.to_base64() for image in self.images],
            "tool_calls": [tool_call.model_dump() for tool_call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
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
        )

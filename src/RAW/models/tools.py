from pydantic import BaseModel
from typing import List, Union, Callable, Generator, AsyncGenerator, Awaitable, Any, Dict, Optional

class ToolParam(BaseModel):
    name: str
    type: str
    description: str
    enums: List[str] = []
    required: bool = False
    items: Optional[Dict[str, Any]] = None
    # Extra JSON Schema keywords merged into the property as-is, e.g.
    # {"properties": {...}, "required": [...]} for objects or {"minimum": 0}.
    json_schema: Optional[Dict[str, Any]] = None

    def to_property(self):
        prop = {
            "type": self.type,
            "description": self.description,
        }
        if self.enums:
            prop["enum"] = self.enums
        if self.items:
            prop["items"] = self.items
        if self.json_schema:
            prop.update(self.json_schema)
        return prop


class Tool(BaseModel):
    name: str
    description: str
    parameters: List['ToolParam']
    function: Union[
        Callable[..., str],
        Callable[..., Awaitable[str]],
        Callable[..., Generator[Any, None, str]],
        Callable[..., AsyncGenerator[str, None]],
    ]

    def parameters_schema(self) -> Dict[str, Any]:
        properties = {}
        required_fields = []

        for param in self.parameters:
            properties[param.name] = param.to_property()
            if param.required:
                required_fields.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required_fields,
        }

    def to_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }
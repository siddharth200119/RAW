from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from .llm_capability import LLMCapability


class LLMInfo(BaseModel):
    model_name: str
    provider: str
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    capabilities: List[LLMCapability] = []
    metadata: Dict[str, Any] = {}

    def to_string(self) -> str:
        caps = ", ".join(c.name for c in self.capabilities) if self.capabilities else "none"
        parts = [
            f"provider={self.provider}",
            f"model={self.model_name}",
            f"capabilities=[{caps}]",
        ]
        if self.context_window is not None:
            parts.append(f"context_window={self.context_window}")
        if self.max_tokens is not None:
            parts.append(f"max_tokens={self.max_tokens}")
        return " | ".join(parts)

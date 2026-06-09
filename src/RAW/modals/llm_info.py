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

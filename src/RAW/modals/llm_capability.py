from enum import Enum, auto


class LLMCapability(Enum):
    """Capabilities that an LLM may support"""
    COMPLETION = auto()
    CHAT = auto()
    TOOLS = auto()
    VISION = auto()
    EMBEDDING = auto()
    STREAMING = auto()

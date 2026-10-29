from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from .models import Tool, Message, Image, LLMCapability, Skill, discover_skills, bundled_skills_dir
from .deciders import BaseDecider, OpenRouterDecider
from .utils import logger
from .utils.metrics import Metrics, NullMetrics, InMemoryMetrics, OpenTelemetryMetrics

try:
    __version__ = version("raw-agents")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Agentic exports need the `raw-agents[agentic]` extra, so they are imported on
# first access instead of at package import.
_AGENTIC_EXPORTS = {
    "Agent": ".agent",
    "agent_tool": ".subagent",
    "GeminiLLM": ".llms",
    "GeminiOptions": ".llms",
    "BaseLLM": ".llms",
}


def __getattr__(name):
    module = _AGENTIC_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)


__all__ = [
    "Agent",
    "agent_tool",
    "GeminiLLM",
    "GeminiOptions",
    "BaseLLM",
    "BaseDecider",
    "OpenRouterDecider",
    "Tool",
    "Message",
    "Image",
    "LLMCapability",
    "Skill",
    "discover_skills",
    "bundled_skills_dir",
    "Metrics",
    "NullMetrics",
    "InMemoryMetrics",
    "OpenTelemetryMetrics",
    "logger"
]

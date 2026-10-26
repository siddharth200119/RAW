from pydantic import BaseModel
from typing import Dict, Optional


class Usage(BaseModel):
    """
    Token usage for one LLM call (or a sum of calls).

    ``input_tokens`` is the *uncached* part of the prompt, so the full prompt
    size is ``input_tokens + cache_read_tokens + cache_write_tokens``. This
    matches the Anthropic semantics; OpenAI-style usage is normalised to it.
    """
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None

    def __add__(self, other: "Usage") -> "Usage":
        def _sum(a: Optional[int], b: Optional[int]) -> Optional[int]:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return Usage(
            input_tokens=_sum(self.input_tokens, other.input_tokens),
            output_tokens=_sum(self.output_tokens, other.output_tokens),
            cache_read_tokens=_sum(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=_sum(self.cache_write_tokens, other.cache_write_tokens),
        )

    @classmethod
    def from_openai(cls, data: Optional[Dict]) -> Optional["Usage"]:
        """Parse an OpenAI-compatible ``usage`` object (OpenAI, vLLM, Groq)."""
        if not data:
            return None
        prompt_tokens = data.get("prompt_tokens")
        cached = (data.get("prompt_tokens_details") or {}).get("cached_tokens")
        if prompt_tokens is not None and cached:
            prompt_tokens -= cached
        return cls(
            input_tokens=prompt_tokens,
            output_tokens=data.get("completion_tokens"),
            cache_read_tokens=cached,
        )

    @classmethod
    def from_anthropic(cls, data: Optional[Dict]) -> Optional["Usage"]:
        """Parse an Anthropic Messages API ``usage`` object."""
        if not data:
            return None
        return cls(
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            cache_read_tokens=data.get("cache_read_input_tokens"),
            cache_write_tokens=data.get("cache_creation_input_tokens"),
        )

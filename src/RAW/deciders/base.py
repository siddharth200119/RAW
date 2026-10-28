from abc import ABC
from typing import Any, Dict, List, Mapping, Union
from RAW.models import Decision, DecisionQuestion

# What the decider evaluates: a string, a JSON object, or a list of context items
# (strings, objects, or RAW Images).
DecisionState = Union[str, Dict[str, Any], List[Any]]


class BaseDecider(ABC):
    """A decision model: answers typed questions with probabilities, never generated text."""

    model: str

    async def decide(
        self,
        state: DecisionState,
        questions: Mapping[str, DecisionQuestion]
    ) -> Decision:
        raise NotImplementedError("This decider does not implement the decide method.")

    async def stop(self) -> None:
        raise NotImplementedError("This decider does not implement the stop method.")

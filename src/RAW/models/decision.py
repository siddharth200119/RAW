"""Typed questions and answers for decision models (OpenRouter Decisions API).

Decision models such as TypeSafe's Jev answer typed questions about a ``state``
with probabilities instead of generated text. Field names match the API.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from .usage import Usage

# Instructions and criteria accept a plain string, or a JSON object or array of structured guidance.
Guidance = Union[str, Dict[str, Any], List[Any]]


class NoulCriteria(BaseModel):
    true: Guidance
    false: Guidance


class NoulQuestion(BaseModel):
    """Yes/no question; answered with the probability of yes."""
    type: Literal["noul"] = "noul"
    instructions: Guidance
    criteria: Optional[NoulCriteria] = None


class ChoiceQuestion(BaseModel):
    """Pick one option; ``criteria`` maps each option name to its description."""
    type: Literal["choice"] = "choice"
    instructions: Guidance
    criteria: Dict[str, Optional[Guidance]]


class ScoreQuestion(BaseModel):
    """Place the state on an ordered scale; ``criteria`` lists the levels, lowest first."""
    type: Literal["score"] = "score"
    instructions: Guidance
    criteria: List[Guidance] = Field(min_length=1)


class BoxQuestion(BaseModel):
    """Locate the described target in an image in ``state`` (Jev 1.14+ only)."""
    type: Literal["box"] = "box"
    instructions: Guidance


DecisionQuestion = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion, BoxQuestion],
    Field(discriminator="type"),
]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: Optional[float] = None
    probabilities: Dict[str, float] = {}


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: Optional[float] = None
    probabilities: Dict[str, float] = {}
    legend: Dict[str, Guidance] = {}


class BoxEdges(BaseModel):
    """Edges as fractions of image width (left/right) and height (top/bottom) from the top-left."""
    top: float
    bottom: float
    left: float
    right: float


class BoxAnswer(BaseModel):
    """Box never reports "not found"; pair it with a NoulQuestion to check presence."""
    type: Literal["box"] = "box"
    box: BoxEdges
    confidence: Optional[float] = None


DecisionAnswer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer, BoxAnswer],
    Field(discriminator="type"),
]


class Decision(BaseModel):
    """One decision response: an answer per question name, plus usage."""
    answers: Dict[str, DecisionAnswer]
    usage: Optional[Usage] = None
    id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None

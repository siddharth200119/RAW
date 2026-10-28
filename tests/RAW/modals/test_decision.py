import pytest
from pydantic import TypeAdapter, ValidationError
from RAW.models import (
    BoxAnswer,
    BoxQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    Decision,
    DecisionQuestion,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)

# Response captured from the live API in OpenRouter's Jev tutorial.
TUTORIAL_ANSWERS = {
    "is_bug": {"type": "noul", "noul": 0.96},
    "team": {
        "type": "choice",
        "choice": "payments",
        "confidence": 0.67,
        "probabilities": {"payments": 0.78, "frontend": 0.22, "account": 0},
    },
    "urgency": {
        "type": "score",
        "score": 1.99,
        "confidence": 0.99,
        "probabilities": {"0": 0, "1": 0, "2": 1},
        "legend": {
            "0": "Can wait for the next release",
            "1": "Should be fixed this week",
            "2": "Blocking revenue right now",
        },
    },
}


def test_questions_dump_to_api_shape():
    noul = NoulQuestion(
        instructions="Is this a bug?",
        criteria=NoulCriteria(true="Broken behaviour.", false="A question."),
    )
    assert noul.model_dump() == {
        "type": "noul",
        "instructions": "Is this a bug?",
        "criteria": {"true": "Broken behaviour.", "false": "A question."},
    }
    assert ChoiceQuestion(instructions="Team?", criteria={"payments": "Billing", "other": None}).model_dump() == {
        "type": "choice",
        "instructions": "Team?",
        "criteria": {"payments": "Billing", "other": None},
    }
    assert ScoreQuestion(instructions="Urgency?", criteria=["low", {"level": "high"}]).model_dump()["criteria"] == [
        "low",
        {"level": "high"},
    ]
    assert BoxQuestion(instructions="The red car").model_dump() == {"type": "box", "instructions": "The red car"}


def test_score_question_needs_criteria():
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="Urgency?", criteria=[])


def test_question_union_dispatches_on_type():
    adapter = TypeAdapter(DecisionQuestion)
    assert isinstance(adapter.validate_python({"type": "box", "instructions": "x"}), BoxQuestion)
    assert isinstance(adapter.validate_python({"type": "noul", "instructions": "x"}), NoulQuestion)


def test_decision_parses_tutorial_response():
    decision = Decision(answers=TUTORIAL_ANSWERS)

    assert isinstance(decision.answers["is_bug"], NoulAnswer)
    assert decision.answers["is_bug"].noul == 0.96

    team = decision.answers["team"]
    assert isinstance(team, ChoiceAnswer)
    assert team.choice == "payments"
    assert team.probabilities["account"] == 0

    urgency = decision.answers["urgency"]
    assert isinstance(urgency, ScoreAnswer)
    assert urgency.score == 1.99
    assert urgency.legend["2"] == "Blocking revenue right now"


def test_decision_parses_box_answer():
    decision = Decision(answers={
        "product": {"type": "box", "box": {"top": 0.49, "bottom": 0.89, "left": 0.58, "right": 0.96}, "confidence": 0.93}
    })
    answer = decision.answers["product"]
    assert isinstance(answer, BoxAnswer)
    assert (answer.box.top, answer.box.right) == (0.49, 0.96)


def test_unknown_answer_type_is_rejected():
    with pytest.raises(ValidationError):
        Decision(answers={"x": {"type": "mystery", "value": 1}})


def test_usage_cost_sums():
    total = Usage(input_tokens=10, cost=0.5) + Usage(input_tokens=5, cost=0.25)
    assert total.cost == 0.75
    assert (Usage(input_tokens=1) + Usage(input_tokens=2)).cost is None
    assert (Usage(cost=0.1) + Usage()).cost == 0.1

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from RAW.models import Message, Skill, Tool
from RAW.utils import Logger
from RAW.utils.router import Router

LOGGER = Logger("test_router")

ROUTE_RESULT = {
    "reasoning": "needs weather",
    "selected_tools": ["get_weather"],
    "selected_skills": [],
    "conversation_summary": "User asked about weather.",
}


async def noop():
    pass


def make_llm(response=None, side_effect=None):
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=response, side_effect=side_effect)
    return llm


def make_tool():
    return Tool(name="get_weather", description="Get the weather", parameters=[], function=noop)


def make_skill(**overrides):
    fields = dict(
        name="travel",
        description="Plan trips",
        tags=["trips"],
        triggers=["book a flight"],
        exclusions=["visa advice"],
        system_prompt="You plan trips.",
        workflow=["ask dates"],
        examples=[],
    )
    fields.update(overrides)
    return Skill(**fields)


async def route(router, tools=None, skills=None, text="what's the weather?"):
    return await router.route(
        user_message=Message(role="user", content=text),
        conversation_summary="prior summary",
        user_summary="likes hiking",
        available_tools=tools or [],
        available_skills=skills or [],
    )


def test_requires_llm():
    with pytest.raises(ValueError):
        Router(llm=None, logger=LOGGER)


async def test_parses_json_string_response():
    llm = make_llm(json.dumps(ROUTE_RESULT))
    result = await route(Router(llm, logger=LOGGER), tools=[make_tool()])
    assert result == {**ROUTE_RESULT, "fallback": False}


async def test_accepts_dict_response():
    llm = make_llm(dict(ROUTE_RESULT))
    result = await route(Router(llm, logger=LOGGER), tools=[make_tool()])
    assert result == {**ROUTE_RESULT, "fallback": False}


async def test_prompt_and_schema_sent_to_llm():
    llm = make_llm(json.dumps(ROUTE_RESULT))
    await route(Router(llm, logger=LOGGER), tools=[make_tool()], skills=[make_skill()])

    kwargs = llm.generate.call_args.kwargs
    prompt = kwargs["prompt"]
    assert kwargs["stream"] is False
    assert "- get_weather: Get the weather" in prompt
    assert "- travel: Plan trips" in prompt
    assert "Tags: trips" in prompt
    assert "Triggers: book a flight" in prompt
    assert "Exclusions: visa advice" in prompt
    assert "prior summary" in prompt
    assert "likes hiking" in prompt
    assert "what's the weather?" in prompt
    assert set(kwargs["schema"]["required"]) == set(ROUTE_RESULT)


async def test_empty_tools_skills_and_skill_fields():
    llm = make_llm(json.dumps(ROUTE_RESULT))
    await route(Router(llm, logger=LOGGER), skills=[make_skill(tags=[], triggers=[], exclusions=[])])

    prompt = llm.generate.call_args.kwargs["prompt"]
    assert "No tools available." in prompt
    assert "Tags: none" in prompt


async def test_invalid_response_type_returns_empty_selection():
    llm = make_llm(None)
    result = await route(Router(llm, logger=LOGGER))
    assert result["selected_tools"] == []
    assert result["selected_skills"] == []
    assert result["conversation_summary"] == "prior summary"


async def test_invalid_json_returns_empty_selection():
    llm = make_llm("not json")
    result = await route(Router(llm, logger=LOGGER))
    assert result["selected_tools"] == []
    assert result["conversation_summary"] == "prior summary"


async def test_llm_exception_falls_back_to_all_tools_and_skills():
    # A failed routing call must not leave the agent without tools (P0-7).
    llm = make_llm(side_effect=RuntimeError("provider down"))
    result = await route(Router(llm, logger=LOGGER), tools=[make_tool()], skills=[make_skill()])
    assert llm.generate.await_count == 2  # one retry
    assert "provider down" in result["reasoning"]
    assert result["fallback"] is True
    assert result["selected_tools"] == ["get_weather"]
    assert result["selected_skills"] == ["travel"]
    assert result["conversation_summary"] == "prior summary"


# --- Decider routing ---

from RAW.deciders import BaseDecider  # noqa: E402
from RAW.models import Decision  # noqa: E402


class ScriptedDecider(BaseDecider):
    def __init__(self, probabilities=None, error=None):
        self.model = "test/decider"
        self.probabilities = probabilities or {}
        self.error = error
        self.calls = []

    async def decide(self, state, questions):
        self.calls.append((state, questions))
        if self.error:
            raise self.error
        return Decision(answers={
            name: {"type": "noul", "noul": p}
            for name, p in self.probabilities.items() if name in questions
        })


def make_other_tool():
    return Tool(name="send_email", description="Send an email", parameters=[], function=noop)


def test_requires_exactly_one_of_llm_or_decider():
    with pytest.raises(ValueError):
        Router(llm=make_llm("{}"), decider=ScriptedDecider(), logger=LOGGER)
    with pytest.raises(ValueError):
        Router(logger=LOGGER)


def test_decider_threshold_must_be_a_probability():
    with pytest.raises(ValueError):
        Router(decider=ScriptedDecider(), decider_threshold=1.5, logger=LOGGER)


async def test_decider_selects_by_threshold():
    decider = ScriptedDecider({"tool_0": 0.9, "tool_1": 0.2, "skill_0": 0.5})
    router = Router(decider=decider, logger=LOGGER)
    result = await route(router, tools=[make_tool(), make_other_tool()], skills=[make_skill()])

    assert result["selected_tools"] == ["get_weather"]
    assert result["selected_skills"] == ["travel"]  # threshold is inclusive
    assert result["probabilities"] == {
        "tools": {"get_weather": 0.9, "send_email": 0.2},
        "skills": {"travel": 0.5},
    }
    # Decision models return no text, so the summary is passed through.
    assert result["conversation_summary"] == "prior summary"


async def test_decider_custom_threshold():
    decider = ScriptedDecider({"tool_0": 0.9, "tool_1": 0.6})
    router = Router(decider=decider, decider_threshold=0.8, logger=LOGGER)
    result = await route(router, tools=[make_tool(), make_other_tool()])
    assert result["selected_tools"] == ["get_weather"]


async def test_decider_questions_and_state():
    decider = ScriptedDecider({})
    await route(Router(decider=decider, logger=LOGGER), tools=[make_tool()], skills=[make_skill()])

    state, questions = decider.calls[0]
    assert state == {
        "user_message": "what's the weather?",
        "conversation_summary": "prior summary",
        "user_summary": "likes hiking",
    }
    assert set(questions) == {"tool_0", "skill_0"}
    assert all(q.type == "noul" for q in questions.values())
    assert "`get_weather`" in questions["tool_0"].instructions
    assert "Get the weather" in questions["tool_0"].criteria.true
    assert "book a flight" in questions["skill_0"].criteria.true
    assert "visa advice" in questions["skill_0"].criteria.false


async def test_decider_skipped_without_tools_or_skills():
    decider = ScriptedDecider()
    result = await route(Router(decider=decider, logger=LOGGER))
    assert decider.calls == []
    assert result["selected_tools"] == [] and result["selected_skills"] == []


async def test_decider_missing_answer_is_not_selected():
    decider = ScriptedDecider({"tool_1": 0.99})
    result = await route(Router(decider=decider, logger=LOGGER), tools=[make_tool(), make_other_tool()])
    assert result["selected_tools"] == ["send_email"]
    assert "get_weather" not in result["probabilities"]["tools"]


async def test_decider_error_falls_back_to_all_tools():
    decider = ScriptedDecider(error=RuntimeError("provider down"))
    result = await route(Router(decider=decider, logger=LOGGER), tools=[make_tool()])
    assert "provider down" in result["reasoning"]
    assert result["fallback"] is True
    assert result["selected_tools"] == ["get_weather"]
    assert result["conversation_summary"] == "prior summary"

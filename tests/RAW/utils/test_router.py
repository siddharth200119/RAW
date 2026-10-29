import json
from typing import List

from RAW.agent import Agent
from RAW.llms.base import BaseLLM
from RAW.models import LLMCapability, Message, Skill, Tool
from RAW.utils import Router


class ScriptedLLM(BaseLLM):
    """generate() returns (or raises) the scripted items in order."""

    def __init__(self, outputs: List):
        self.outputs = list(outputs)
        self.generate_calls = 0
        self.capabilities = [LLMCapability.CHAT, LLMCapability.TOOLS]

    async def generate(self, prompt, images=None, schema=None, stream=False):
        self.generate_calls += 1
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output

    async def chat(self, messages, schema=None, stream=False, tools=None):
        return Message(role="assistant", content="done")


def tool(name):
    return Tool(name=name, description=name, parameters=[], function=lambda: name)


def skill(name):
    return Skill(name=name, description=name, system_prompt="", workflow=[], examples=[])


def routed(tools=None, skills=None, summary="", **overrides):
    result = {
        "reasoning": "r",
        "selected_tools": tools or [],
        "selected_skills": skills or [],
        "conversation_summary": summary,
    }
    result.update(overrides)
    return json.dumps(result)


TOOLS = [tool("stock"), tool("po")]
SKILLS = [skill("purchase")]


async def route(router):
    return await router.route(
        user_message=Message(role="user", content="hi"),
        conversation_summary="old summary",
        user_summary="",
        available_tools=TOOLS,
        available_skills=SKILLS,
    )


async def test_failure_after_retry_falls_back_to_all_tools_and_skills():
    llm = ScriptedLLM([RuntimeError("timeout"), RuntimeError("timeout again")])
    result = await route(Router(llm))

    assert llm.generate_calls == 2
    assert result["fallback"] is True
    assert result["selected_tools"] == ["stock", "po"]
    assert result["selected_skills"] == ["purchase"]
    assert result["conversation_summary"] == "old summary"


async def test_retry_recovers_from_one_failure():
    llm = ScriptedLLM([RuntimeError("timeout"), routed(tools=["po"], summary="new")])
    result = await route(Router(llm))

    assert result["fallback"] is False
    assert result["selected_tools"] == ["po"]
    assert result["conversation_summary"] == "new"


async def test_invalid_output_is_retried_then_falls_back():
    llm = ScriptedLLM(["not json", json.dumps({"selected_tools": "po"})])
    result = await route(Router(llm))

    assert result["fallback"] is True
    assert result["selected_tools"] == ["stock", "po"]


async def test_retries_are_configurable():
    llm = ScriptedLLM([RuntimeError("a"), RuntimeError("b"), routed(tools=["stock"])])
    result = await route(Router(llm, retries=2))

    assert llm.generate_calls == 3
    assert result["selected_tools"] == ["stock"]


async def test_unknown_names_are_dropped():
    llm = ScriptedLLM([routed(tools=["po", "made_up_tool"], skills=["purchase", "ghost"])])
    result = await route(Router(llm))

    assert result["selected_tools"] == ["po"]
    assert result["selected_skills"] == ["purchase"]


async def test_empty_selection_is_respected():
    result = await route(Router(ScriptedLLM([routed()])))

    assert result["fallback"] is False
    assert result["selected_tools"] == []


async def test_agent_keeps_its_tools_when_router_fails():
    router = Router(ScriptedLLM([RuntimeError("down"), RuntimeError("down")]))
    agent = Agent(name="t", llm=ScriptedLLM([]), tools=TOOLS, skills=SKILLS, router=router)

    [event async for event in agent("hi")]

    assert [t.name for t in agent.available_tools] == ["stock", "po"]
    assert [s.name for s in agent.available_skills] == ["purchase"]

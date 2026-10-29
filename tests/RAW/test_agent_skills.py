import textwrap
from typing import List, Optional

import pytest

from RAW.agent import Agent
from RAW.llms.base import BaseLLM
from RAW.models import LLMCapability, Message, Skill, Tool, ToolCall
from RAW.utils import Logger

LOGGER = Logger("test_agent_skills")


class RecordingLLM(BaseLLM):
    """Plays back queued responses and records the tool names and messages of every request."""

    def __init__(self, responses: Optional[List[Message]] = None, capabilities=None):
        self.capabilities = capabilities or [LLMCapability.COMPLETION, LLMCapability.TOOLS]
        self.responses = list(responses or [Message(role="assistant", content="done")])
        self.tool_names: List[List[str]] = []
        self.seen_messages: List[List[Message]] = []

    async def chat(self, messages, schema=None, stream=False, tools=None):
        self.tool_names.append([t.name for t in tools or []])
        self.seen_messages.append(list(messages))
        return self.responses.pop(0) if self.responses else Message(role="assistant", content="done")


def call(tool, call_id="c1", /, **arguments):
    return Message(role="assistant", tool_calls=[ToolCall(id=call_id, name=tool, arguments=arguments)])


def final(text="done"):
    return Message(role="assistant", content=text)


def hello_tool():
    return Tool(name="hello", description="says hi", parameters=[], function=lambda: "hi there")


def ping_tool():
    return Tool(name="ping", description="ping", parameters=[], function=lambda: "pong")


def code_skill(name="greeter", with_tool=True):
    return Skill(
        name=name,
        description=f"Greets people ({name})",
        system_prompt=f"FULL INSTRUCTIONS FOR {name}",
        tools=[hello_tool()] if with_tool else [],
    )


async def run(agent, message="go", **kwargs):
    return [chunk async for chunk in agent(message, **kwargs)]


def tool_results(llm_request):
    return [m.content for m in llm_request if m.role == "tool"]


# --- Catalog mode -----------------------------------------------------------------

async def test_catalog_lists_skills_without_their_instructions_or_tools():
    llm = RecordingLLM()
    agent = Agent(name="a", llm=llm, tools=[ping_tool()], skills=[code_skill()], logger=LOGGER)

    await run(agent)

    system = llm.seen_messages[0][0].content
    assert "# Available skills" in system
    assert "- greeter: Greets people (greeter)" in system
    assert "FULL INSTRUCTIONS" not in system
    assert llm.tool_names[0] == ["ping", "load_skill"]


async def test_load_skill_returns_instructions_and_unlocks_the_skills_tools():
    llm = RecordingLLM([call("load_skill", name="greeter"), call("hello", "c2"), final("ok")])
    agent = Agent(name="a", llm=llm, skills=[code_skill()], logger=LOGGER)

    chunks = await run(agent)

    assert chunks[-1]["content"] == "ok"
    assert "hello" not in llm.tool_names[0]
    assert "hello" in llm.tool_names[1] and "hello" in llm.tool_names[2]
    loaded = tool_results(llm.seen_messages[1])
    assert loaded == [agent.skills[0].build_prompt()]
    assert "FULL INSTRUCTIONS FOR greeter" in loaded[0] and "hello()" in loaded[0]
    assert tool_results(llm.seen_messages[2])[-1] == "hi there"


async def test_skill_tool_is_not_callable_before_the_skill_is_loaded():
    llm = RecordingLLM([call("hello"), final()])
    agent = Agent(name="a", llm=llm, skills=[code_skill()], logger=LOGGER)

    await run(agent)

    assert "not available" in tool_results(llm.seen_messages[1])[0]


async def test_unknown_skill_name_is_reported_to_the_model():
    llm = RecordingLLM([call("load_skill", name="nope"), final()])
    agent = Agent(name="a", llm=llm, skills=[code_skill()], logger=LOGGER)

    chunks = await run(agent)

    result = tool_results(llm.seen_messages[1])[0]
    # Rejected by argument validation (the name enum) before load_skill runs.
    assert "'nope' is not one of" in result and "greeter" in result
    assert chunks[-1]["content"] == "done"


async def test_loaded_skills_stay_loaded_and_leave_the_catalog():
    llm = RecordingLLM([call("load_skill", name="a"), final("one"), final("two")])
    agent = Agent(name="x", llm=llm, skills=[code_skill("a"), code_skill("b", with_tool=False)], logger=LOGGER)

    await run(agent)
    await run(agent, "again")

    system = llm.seen_messages[-1][0].content
    assert "- b:" in system and "- a:" not in system
    assert "hello" in llm.tool_names[-1]


async def test_load_skill_tool_lists_skill_names():
    agent = Agent(name="a", llm=RecordingLLM(), skills=[code_skill("a"), code_skill("b")], logger=LOGGER)
    load = next(t for t in agent.available_tools if t.name == "load_skill")
    assert load.parameters[0].enums == ["a", "b"]


# --- Inline, router and override ----------------------------------------------------

async def test_inline_mode_puts_everything_in_the_prompt():
    llm = RecordingLLM()
    agent = Agent(name="a", llm=llm, skills=[code_skill()], skill_mode="inline", logger=LOGGER)

    await run(agent)

    assert "FULL INSTRUCTIONS FOR greeter" in llm.seen_messages[0][0].content
    assert "Available skills" not in llm.seen_messages[0][0].content
    assert llm.tool_names[0] == ["hello"]


class FakeRouter:
    def __init__(self, skills):
        self.skills = skills

    async def route(self, **kwargs):
        return {"selected_tools": [], "selected_skills": self.skills, "conversation_summary": ""}


async def test_router_selected_skills_are_inline_and_there_is_no_load_skill():
    llm = RecordingLLM()
    agent = Agent(
        name="a", llm=llm, skills=[code_skill("a"), code_skill("b", with_tool=False)],
        router=FakeRouter(["a"]), logger=LOGGER,
    )

    await run(agent)

    system = llm.seen_messages[0][0].content
    assert "FULL INSTRUCTIONS FOR a" in system and "FOR b" not in system
    assert "Available skills" not in system
    assert llm.tool_names[0] == ["hello"]


async def test_skill_argument_overrides_and_loads_inline():
    llm = RecordingLLM()
    agent = Agent(name="a", llm=llm, skills=[code_skill("a"), code_skill("b")], logger=LOGGER)

    await run(agent, skill=agent.skills[1])

    assert "FULL INSTRUCTIONS FOR b" in llm.seen_messages[0][0].content
    assert llm.tool_names[0] == ["hello"]


async def test_llm_without_tool_support_falls_back_to_inline():
    llm = RecordingLLM(capabilities=[LLMCapability.COMPLETION])
    agent = Agent(name="a", llm=llm, skills=[code_skill()], logger=LOGGER)

    assert agent.skill_mode == "inline"
    await run(agent)
    assert "FULL INSTRUCTIONS FOR greeter" in llm.seen_messages[0][0].content


# --- Folders --------------------------------------------------------------------------

def make_skill_folder(root, name="pdf", body="Read reference.md first.", tools=False):
    folder = root / name
    (folder / "scripts").mkdir(parents=True)
    (folder / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: Work with PDF files.
        license: MIT
        ---
        {body}
    """))
    (folder / "reference.md").write_text("REFERENCE TEXT")
    (folder / "scripts" / "split.py").write_text("print('split')")
    if tools:
        (folder / "tools.py").write_text(textwrap.dedent("""\
            from RAW.models import Tool
            TOOLS = [Tool(name="hello", description="hi", parameters=[], function=lambda: "hi there")]
        """))
    return folder


async def test_skills_dir_loads_claude_style_skills(tmp_path):
    make_skill_folder(tmp_path / "skills", "pdf")
    llm = RecordingLLM()
    agent = Agent(name="a", llm=llm, skills_dir=tmp_path / "skills", logger=LOGGER)

    await run(agent)

    assert [s.name for s in agent.skills] == ["pdf"]
    assert "- pdf: Work with PDF files." in llm.seen_messages[0][0].content


async def test_skills_dir_combines_with_skills_and_accepts_a_list(tmp_path):
    make_skill_folder(tmp_path / "one", "pdf")
    make_skill_folder(tmp_path / "two", "csv")
    agent = Agent(
        name="a", llm=RecordingLLM(), skills=[code_skill()],
        skills_dir=[str(tmp_path / "one"), tmp_path / "two"], logger=LOGGER,
    )
    assert [s.name for s in agent.skills] == ["greeter", "pdf", "csv"]


async def test_skill_folder_tools_py_needs_opt_in(tmp_path):
    make_skill_folder(tmp_path / "skills", "pdf", tools=True)
    off = Agent(name="a", llm=RecordingLLM(), skills_dir=tmp_path / "skills", skill_mode="inline", logger=LOGGER)
    on = Agent(
        name="a", llm=RecordingLLM(), skills_dir=tmp_path / "skills", skill_mode="inline",
        load_skill_tools=True, logger=LOGGER,
    )
    assert "hello" not in [t.name for t in off.available_tools]
    assert "hello" in [t.name for t in on.available_tools]


async def test_read_skill_resource_serves_files_and_blocks_escapes(tmp_path):
    make_skill_folder(tmp_path / "skills", "pdf")
    (tmp_path / "secret.txt").write_text("SECRET")
    llm = RecordingLLM([
        call("load_skill", "c1", name="pdf"),
        call("read_skill_resource", "c2", skill="pdf", path="reference.md"),
        call("read_skill_resource", "c3", skill="pdf", path="../../secret.txt"),
        final(),
    ])
    agent = Agent(name="a", llm=llm, skills_dir=tmp_path / "skills", logger=LOGGER)

    await run(agent)

    assert "read_skill_resource" in llm.tool_names[0]
    loaded = tool_results(llm.seen_messages[1])[0]
    assert "Paths in the instructions above are relative to the skill folder" in loaded
    assert "  - scripts/split.py" in loaded
    results = tool_results(llm.seen_messages[3])
    assert results[1] == "REFERENCE TEXT"
    assert "outside the skill folder" in results[2] and "SECRET" not in results[2]


async def test_read_skill_resource_is_absent_when_no_skill_has_files():
    agent = Agent(name="a", llm=RecordingLLM(), skills=[code_skill()], logger=LOGGER)
    assert "read_skill_resource" not in [t.name for t in agent.available_tools]


# --- Validation -------------------------------------------------------------------------

def test_invalid_configuration_raises():
    with pytest.raises(ValueError, match="unique"):
        Agent(name="a", llm=RecordingLLM(), skills=[code_skill(), code_skill()], logger=LOGGER)
    with pytest.raises(ValueError, match="reserved"):
        Agent(
            name="a", llm=RecordingLLM(), logger=LOGGER,
            tools=[Tool(name="load_skill", description="x", parameters=[], function=lambda: "")],
        )
    with pytest.raises(ValueError, match="skill_mode"):
        Agent(name="a", llm=RecordingLLM(), skill_mode="nope", logger=LOGGER)

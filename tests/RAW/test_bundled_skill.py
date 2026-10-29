"""
Guards the documentation that ships in the bundled skill: it must load, link only to files
that exist, and its Python examples must still match the real API.
"""
import ast
import inspect
import re
from pathlib import Path
from typing import Dict, List, Set

import pytest
from pydantic import BaseModel

import RAW
from RAW.models import Skill, discover_skills, bundled_skills_dir

SKILL_DIR = bundled_skills_dir() / "raw-agents"
DOCS = sorted(SKILL_DIR.rglob("*.md"))


def python_blocks() -> List[tuple]:
    blocks = []
    for doc in DOCS:
        for i, match in enumerate(re.finditer(r"```python\n(.*?)```", doc.read_text(), re.S)):
            blocks.append(pytest.param(match.group(1), id=f"{doc.relative_to(SKILL_DIR)}#{i}"))
    return blocks


def test_the_skill_loads_in_claude_format():
    skill = Skill.from_directory(SKILL_DIR)

    assert skill.name == "raw-agents"
    assert skill.description.startswith("Use when")
    assert "raw-agents" in skill.description and "RAW" in skill.description
    assert [s.name for s in discover_skills(bundled_skills_dir())] == ["raw-agents"]
    assert skill.metadata.get("license") == "MIT"
    assert len(skill.system_prompt.splitlines()) < 200  # the main file stays short; details live in references/


def test_every_referenced_file_exists_and_every_reference_is_linked():
    main = (SKILL_DIR / "SKILL.md").read_text()
    linked = set(re.findall(r"`(references/[\w.-]+\.md)`", main))
    on_disk = {p.relative_to(SKILL_DIR).as_posix() for p in (SKILL_DIR / "references").glob("*.md")}

    assert linked <= on_disk, f"SKILL.md points to missing files: {linked - on_disk}"
    assert on_disk <= linked, f"references never mentioned in SKILL.md: {on_disk - linked}"


def test_the_skill_ships_no_junk():
    names = {p.name for p in SKILL_DIR.rglob("*") if p.is_file()}
    assert not any(n.endswith((".pyc", ".bak")) or n.startswith(".") for n in names)


@pytest.mark.parametrize("code", python_blocks())
def test_python_examples_are_valid_and_their_imports_resolve(code):
    # Top level `await` is fine in the examples.
    tree = compile(code, "<skill>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT | ast.PyCF_ONLY_AST)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<import>", "exec"), {})


def _known_callables() -> Dict[str, object]:
    from RAW.agent import Agent
    from RAW.deciders import OpenRouterDecider
    from RAW.llms import (AnthropicLLM, AnthropicOptions, GeminiLLM, GroqLLM, OpenAILLM, OpenAIOptions, VLLM)
    from RAW.models import (ChoiceQuestion, File, Message, NoulCriteria, NoulQuestion, ScoreQuestion, Tool, ToolParam)
    from RAW.subagent import agent_tool
    from RAW.utils import Logger, Router
    from RAW.utils.storage import LocalStorage, S3Storage

    return {c.__name__: c for c in (
        Agent, OpenRouterDecider, AnthropicLLM, AnthropicOptions, GeminiLLM, GroqLLM, OpenAILLM, OpenAIOptions,
        VLLM, ChoiceQuestion, File, Message, NoulCriteria, NoulQuestion, ScoreQuestion, Tool, ToolParam,
        agent_tool, Logger, Router, LocalStorage, S3Storage, Skill,
    )}


def _accepted_keywords(target) -> Set[str]:
    if inspect.isclass(target) and issubclass(target, BaseModel):
        return set(target.model_fields)
    return set(inspect.signature(target).parameters)


@pytest.mark.parametrize("code", python_blocks())
def test_python_examples_only_use_real_parameters(code):
    known = _known_callables()
    tree = compile(code, "<skill>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT | ast.PyCF_ONLY_AST)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in known:
            used = {kw.arg for kw in node.keywords if kw.arg}
            unknown = used - _accepted_keywords(known[node.func.id])
            assert not unknown, f"{node.func.id}(...) has no parameter(s) {sorted(unknown)}"


def test_documented_agent_parameters_exist():
    text = (SKILL_DIR / "references" / "agent.md").read_text()
    block = re.search(r"```python\nfrom RAW import Agent\n\nagent = Agent\((.*?)\n\)\n```", text, re.S).group(1)
    documented = set(re.findall(r"^\s+(\w+)=", block, re.M))
    documented |= set(re.findall(r",\s*(\w+)=", block))
    real = set(inspect.signature(RAW.Agent.__init__).parameters) - {"self"}

    assert documented <= real
    # Every constructor parameter is documented.
    assert real <= documented, f"undocumented Agent parameters: {real - documented}"


def test_documented_env_vars_and_defaults_match_the_code():
    providers = (SKILL_DIR / "references" / "providers.md").read_text()
    from RAW.llms import AnthropicLLM, GeminiLLM, OpenAILLM

    for cls, var in ((AnthropicLLM, "ANTHROPIC_API_KEY"), (OpenAILLM, "OPENAI_API_KEY"), (GeminiLLM, "GEMINI_API_KEY")):
        assert var in providers
        assert var in Path(inspect.getsourcefile(cls)).read_text()
    for cls in (AnthropicLLM, OpenAILLM, GeminiLLM):
        default_model = inspect.signature(cls.__init__).parameters["model"].default
        assert f'model="{default_model}"' in providers

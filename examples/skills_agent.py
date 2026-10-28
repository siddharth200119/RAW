"""
Using skills: a Claude-style skills folder, loaded on demand.

Every folder under ``examples/skills`` with a ``SKILL.md`` becomes a skill, so you can also
point ``skills_dir`` at skills written for Claude (for example ``~/.claude/skills``). The system
prompt lists each skill's name and description; the model calls ``load_skill`` to read the full
instructions, which also unlocks the tools from the skill's ``tools.py``. Reference files such as
``references/template.md`` are read with ``read_skill_resource``.

Needs a running vLLM server (see main.py).

    uv run python examples/skills_agent.py
"""
import asyncio
from pathlib import Path

from RAW import Agent
from RAW.llms import VLLM

SKILLS = Path(__file__).parent / "skills"


async def main() -> None:
    llm = VLLM(model="main-model", base_url="http://localhost:8000")

    agent = Agent(
        name="Assistant",
        llm=llm,
        skills_dir=SKILLS,
        load_skill_tools=True,    # imports tools.py from skill folders: only enable for skills you trust
        skill_mode="catalog",     # "inline" puts every skill's full prompt in the system prompt instead
    )

    transcript = "Ana: ship v2 on Friday. Raj: I'll update the docs by Thursday. Ana: we still need a rollback plan."
    async for chunk in agent(f"Write minutes for this meeting:\n{transcript}"):
        content = chunk["content"]
        if isinstance(content, dict) and "tool_call" in content:
            print(f"[tool] {content['tool_call']} {content['arguments']}")
        elif isinstance(content, str):
            print(content)


if __name__ == "__main__":
    asyncio.run(main())

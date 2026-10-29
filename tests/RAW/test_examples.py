"""Keeps the example skills loadable, so they cannot rot unnoticed."""
from pathlib import Path

from RAW.models import discover_skills

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_example_skills_load_with_their_tools():
    skills = discover_skills(EXAMPLES / "skills", load_tools=True)

    assert [s.name for s in skills] == ["meeting-notes"]
    skill = skills[0]
    assert skill.resources() == ["references/template.md"]
    assert [t.name for t in skill.tools] == ["count_words"]
    assert skill.tools[0].function("one two three") == "3"
    assert "## Action items" in skill.read_resource("references/template.md")

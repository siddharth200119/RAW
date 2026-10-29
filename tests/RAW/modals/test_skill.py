import os
import textwrap

import pytest

from RAW.models import Skill, Tool, discover_skills
from RAW.models.skill import MAX_RESOURCE_CHARS


def md(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def write_skill(root, name, text, **files):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(md(text), encoding="utf-8")
    for relative, content in files.items():
        target = folder / relative.replace("__", "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return folder


# --- Claude-format frontmatter ------------------------------------------------

def test_plain_claude_skill():
    skill = Skill.from_markdown(md("""
        ---
        name: pdf
        description: Use this skill whenever the user wants to work with PDF files.
        license: Proprietary. LICENSE.txt has complete terms
        ---

        # PDF Processing Guide

        Use pypdf to read pages.
    """))
    assert skill.name == "pdf"
    assert skill.description == "Use this skill whenever the user wants to work with PDF files."
    assert skill.system_prompt.startswith("# PDF Processing Guide")
    assert skill.metadata == {"license": "Proprietary. LICENSE.txt has complete terms"}
    assert skill.workflow == [] and skill.examples == [] and skill.tools == []
    assert "## Instructions" in skill.build_prompt()


def test_nested_metadata_allowed_tools_and_folded_description():
    skill = Skill.from_markdown(md("""
        ---
        name: report-writer
        description: >
          Writes weekly reports.
          Use when the user asks for a status update.
        allowed-tools: Read, Grep, Bash(git log:*)
        metadata:
          author: Jane Doe
          version: "2.1"
          category: writing
        argument-hint: "[week]"
        ---
        Body
    """))
    assert skill.description == "Writes weekly reports. Use when the user asks for a status update."
    assert skill.author == "Jane Doe"
    assert skill.version == "2.1"
    assert skill.metadata["allowed-tools"] == "Read, Grep, Bash(git log:*)"
    assert skill.metadata["metadata"] == {"author": "Jane Doe", "version": "2.1", "category": "writing"}
    assert skill.metadata["argument-hint"] == "[week]"


def test_quotes_colons_comments_and_lists():
    skill = Skill.from_markdown(md("""
        ---
        # a comment line
        name: "data-tool"
        description: 'It''s for data: CSV, JSON'   # trailing comment
        tags: [csv, "json"]
        triggers:
          - load a file
          - "summarize: a table"
        exclusions:
        - images
        workflow: |
          step one
          step two
        ---
        x
    """))
    assert skill.name == "data-tool"
    assert skill.description == "It's for data: CSV, JSON"
    assert skill.tags == ["csv", "json"]
    assert skill.triggers == ["load a file", "summarize: a table"]
    assert skill.exclusions == ["images"]
    assert skill.workflow == ["step one\nstep two"]


def test_no_frontmatter_uses_folder_name_and_first_paragraph(tmp_path):
    folder = write_skill(tmp_path, "release-notes", """
        # Release notes

        Summarize merged pull requests into notes.

        ## Steps
        1. list PRs
    """)
    skill = Skill.from_directory(folder)
    assert skill.name == "release-notes"
    assert skill.description == "Summarize merged pull requests into notes."
    assert "## Steps" in skill.system_prompt


def test_missing_description_falls_back_to_body_and_empty_skill_fails():
    skill = Skill.from_markdown("---\nname: a\n---\nDoes a thing.\n")
    assert skill.description == "Does a thing."
    with pytest.raises(ValueError, match="description"):
        Skill.from_markdown("---\nname: a\n---\n")


def test_windows_line_endings_and_bom():
    skill = Skill.from_markdown("﻿---\r\nname: win\r\ndescription: Works.\r\n---\r\nBody\r\n")
    assert skill.name == "win" and skill.description == "Works."


def test_unknown_and_malformed_lines_do_not_block_loading():
    skill = Skill.from_markdown(md("""
        ---
        name: odd
        description: Still loads
        this line is not yaml
        hooks:
          pre:
            deep: value
        ---
        Body
    """))
    assert skill.name == "odd"
    assert "hooks" in skill.metadata


def test_unclosed_frontmatter_is_an_error():
    with pytest.raises(ValueError, match="closing"):
        Skill.from_markdown("---\nname: a\ndescription: b\n")


def test_existing_python_skill_still_works():
    skill = Skill(name="s", description="d", system_prompt="p", workflow=["w"], examples=[("u", "a")])
    prompt = skill.build_prompt()
    assert "1. w" in prompt and "User: u" in prompt
    assert "## Resources" not in prompt


# --- Folders, resources, tools --------------------------------------------------

def test_resources_are_listed_and_readable(tmp_path):
    folder = write_skill(
        tmp_path, "pdf", "---\nname: pdf\ndescription: d\n---\nSee reference.md",
        **{"reference.md": "REF", "scripts__run.py": "print(1)", ".hidden": "x", "tools.py": "TOOLS = []"},
    )
    (folder / "__pycache__").mkdir()
    (folder / "__pycache__" / "x.pyc").write_text("x")
    skill = Skill.from_directory(folder)

    assert skill.resources() == ["reference.md", "scripts/run.py"]
    assert skill.read_resource("reference.md") == "REF"
    assert skill.read_resource("scripts/run.py") == "print(1)"
    prompt = skill.build_prompt()
    assert "## Resources" in prompt and "  - scripts/run.py" in prompt and "read_skill_resource" in prompt


def test_read_resource_refuses_paths_outside_the_folder(tmp_path):
    folder = write_skill(tmp_path, "s", "---\nname: s\ndescription: d\n---\nx")
    (tmp_path / "secret.txt").write_text("SECRET")
    (folder / "link.txt").symlink_to(tmp_path / "secret.txt")
    skill = Skill.from_directory(folder)

    for bad in ["../secret.txt", str(tmp_path / "secret.txt"), "link.txt", "sub/../../secret.txt"]:
        with pytest.raises(ValueError, match="outside"):
            skill.read_resource(bad)
    with pytest.raises(ValueError, match="no resource"):
        skill.read_resource("missing.md")


def test_read_resource_rejects_binary_and_truncates_long_files(tmp_path):
    folder = write_skill(tmp_path, "s", "---\nname: s\ndescription: d\n---\nx", **{"big.md": "a" * (MAX_RESOURCE_CHARS + 10)})
    (folder / "img.bin").write_bytes(b"\xff\xfe\x00\x80")
    skill = Skill.from_directory(folder)

    assert "not a text file" in pytest.raises(ValueError, skill.read_resource, "img.bin").value.args[0]
    text = skill.read_resource("big.md")
    assert len(text) < MAX_RESOURCE_CHARS + 100 and "truncated" in text


def test_code_skill_has_no_resources():
    skill = Skill(name="s", description="d", system_prompt="p")
    assert skill.resources() == []
    with pytest.raises(ValueError):
        skill.read_resource("x")


TOOLS_PY = """
from RAW.models import Tool

def hello():
    return "hi"

TOOLS = [Tool(name="hello", description="says hi", parameters=[], function=hello)]
"""


def test_tools_py_runs_only_when_asked(tmp_path):
    marker = tmp_path / "ran"
    folder = write_skill(
        tmp_path, "s", "---\nname: s\ndescription: d\n---\nx",
        **{"tools.py": TOOLS_PY + f"\nopen({str(marker)!r}, 'w').write('1')\n"},
    )
    assert Skill.from_directory(folder).tools == []
    assert not marker.exists()

    skill = Skill.from_directory(folder, load_tools=True)
    assert marker.exists()
    assert [t.name for t in skill.tools] == ["hello"]
    assert isinstance(skill.tools[0], Tool)


def test_tools_py_must_define_tools(tmp_path):
    folder = write_skill(tmp_path, "s", "---\nname: s\ndescription: d\n---\nx", **{"tools.py": "X = 1"})
    with pytest.raises(ValueError, match="TOOLS"):
        Skill.from_directory(folder, load_tools=True)


# --- Discovery ------------------------------------------------------------------

def test_discover_parent_folder_single_skill_and_lists(tmp_path):
    write_skill(tmp_path / "all", "b", "---\nname: b\ndescription: d\n---\nx")
    write_skill(tmp_path / "all", "a", "---\nname: a\ndescription: d\n---\nx")
    (tmp_path / "all" / "not-a-skill").mkdir()
    single = write_skill(tmp_path / "other", "c", "---\nname: c\ndescription: d\n---\nx")

    assert [s.name for s in discover_skills(tmp_path / "all")] == ["a", "b"]
    assert [s.name for s in discover_skills(str(single))] == ["c"]
    assert [s.name for s in discover_skills([tmp_path / "all", single])] == ["a", "b", "c"]


def test_discover_errors_name_the_problem(tmp_path):
    write_skill(tmp_path / "one", "x", "---\nname: dup\ndescription: d\n---\nx")
    write_skill(tmp_path / "two", "y", "---\nname: dup\ndescription: d\n---\nx")
    write_skill(tmp_path / "bad", "z", "---\nname: z\n")

    with pytest.raises(ValueError, match="Duplicate skill name 'dup'"):
        discover_skills([tmp_path / "one", tmp_path / "two"])
    with pytest.raises(ValueError, match="z"):
        discover_skills(tmp_path / "bad")
    with pytest.raises(ValueError, match="not a directory"):
        discover_skills(tmp_path / "missing")
    assert os.path.isdir(tmp_path / "one")

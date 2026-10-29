from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel
from .tools import Tool

SKILL_FILE = "SKILL.md"
TOOLS_FILE = "tools.py"
MAX_RESOURCE_CHARS = 50_000
MAX_LISTED_RESOURCES = 50


class Skill(BaseModel):
    version: str = "1.0"
    author: Optional[str] = None
    name: str
    description: str
    tags: List[str] = []
    triggers: List[str] = []
    exclusions: List[str] = []
    system_prompt: str
    workflow: List[str] = []
    examples: List[tuple[str, str]] = []
    tools: List[Tool] = []
    # Directory the skill was loaded from; set by from_directory, None for skills built in code.
    path: Optional[Path] = None

    # Frontmatter keys RAW has no field for (license, allowed-tools, ...), kept as written.
    metadata: Dict[str, Any] = {}

    @classmethod
    def from_markdown(cls, text: str, path: Optional[Path] = None, name: Optional[str] = None) -> "Skill":
        """
        Build a skill from ``SKILL.md`` text in the Claude skills format: optional ``---``
        frontmatter with ``name`` and ``description`` followed by the instructions in Markdown.

        Loading is lenient so Claude-authored skills work as they are: ``name`` falls back to
        ``name`` (the folder name), ``description`` to the first paragraph of the instructions,
        and frontmatter without a closing line is the only structural error. Keys RAW does not
        use end up in ``metadata``. RAW's own optional keys are ``version``, ``author``,
        ``tags``, ``triggers``, ``exclusions`` and ``workflow``.
        """
        meta, body = _parse_frontmatter(text)
        nested = meta.get("metadata")
        extra: Dict[str, Any] = nested if isinstance(nested, dict) else {}
        skill_name = str(meta.get("name") or name or "").strip()
        if not skill_name:
            raise ValueError("A skill needs a name: set `name` in the frontmatter.")
        description = str(meta.get("description") or "").strip() or _first_paragraph(body)
        if not description:
            raise ValueError(f"Skill '{skill_name}' needs a `description` in the frontmatter.")
        fields: Dict[str, Any] = {
            key: _as_list(meta[key]) for key in ("tags", "triggers", "exclusions", "workflow") if key in meta
        }
        for key in ("version", "author"):
            value = meta.get(key, extra.get(key))
            if value is not None:
                fields[key] = str(value)
        known = {"name", "description", "version", "author", "tags", "triggers", "exclusions", "workflow"}
        return cls(
            name=skill_name,
            description=description,
            system_prompt=body.strip(),
            path=path,
            metadata={k: v for k, v in meta.items() if k not in known},
            **fields,
        )

    @classmethod
    def from_directory(cls, path: Union[str, Path], load_tools: bool = False) -> "Skill":
        """
        Load a skill folder: ``SKILL.md`` plus optional reference files.

        ``tools.py`` is only imported when ``load_tools=True``, because importing it runs
        its code. It must define ``TOOLS``, a list of :class:`Tool`.
        """
        root = Path(path)
        skill_file = root / SKILL_FILE
        if not skill_file.is_file():
            raise ValueError(f"No {SKILL_FILE} in {root}")
        skill = cls.from_markdown(
            skill_file.read_text(encoding="utf-8"), path=root, name=root.resolve().name
        )
        tools_file = root / TOOLS_FILE
        if load_tools and tools_file.is_file():
            skill.tools = _load_tools_file(tools_file)
        return skill

    def resources(self) -> List[str]:
        """Files next to ``SKILL.md`` the instructions may point to, relative to the skill folder."""
        if self.path is None or not self.path.is_dir():
            return []
        found: List[str] = []
        for file in sorted(self.path.rglob("*")):
            relative = file.relative_to(self.path)
            if not file.is_file() or "__pycache__" in relative.parts or any(
                part.startswith(".") for part in relative.parts
            ):
                continue
            if relative.as_posix() in (SKILL_FILE, TOOLS_FILE):
                continue
            found.append(relative.as_posix())
        return found

    def read_resource(self, relative_path: str) -> str:
        """Read a text file inside the skill folder; anything that resolves outside it is refused."""
        if self.path is None:
            raise ValueError(f"Skill '{self.name}' was not loaded from a directory and has no resources.")
        root = self.path.resolve()
        target = (root / relative_path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"'{relative_path}' is outside the skill folder.")
        if not target.is_file():
            raise ValueError(f"Skill '{self.name}' has no resource '{relative_path}'.")
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"'{relative_path}' is not a text file.")
        if len(content) > MAX_RESOURCE_CHARS:
            content = content[:MAX_RESOURCE_CHARS] + f"\n[... truncated at {MAX_RESOURCE_CHARS} characters ...]"
        return content

    def build_prompt(self) -> str:
        """
        Converts the skill into a structured prompt block that an LLM can
        consume directly as a system prompt or injected context.
        """
        lines: List[str] = []

        # --- Identity ---
        lines.append(f"# Skill: {self.name}")
        if self.author:
            lines.append(f"Author: {self.author}  |  Version: {self.version}")
        lines.append("")
        lines.append("## Description")
        lines.append(self.description)

        # --- Behavioural guardrails ---
        if self.triggers:
            lines.append("")
            lines.append("## When to activate")
            lines.append("Apply this skill when the user's request involves any of the following:")
            for trigger in self.triggers:
                lines.append(f"  - {trigger}")

        if self.exclusions:
            lines.append("")
            lines.append("## When NOT to activate")
            lines.append("Do NOT apply this skill when:")
            for exclusion in self.exclusions:
                lines.append(f"  - {exclusion}")

        if self.tags:
            lines.append("")
            lines.append("## Tags")
            lines.append(", ".join(self.tags))

        # --- Core instructions ---
        lines.append("")
        lines.append("## Instructions")
        lines.append(self.system_prompt)

        # --- Workflow ---
        if self.workflow:
            lines.append("")
            lines.append("## Workflow")
            lines.append("Follow these steps in order:")
            for i, step in enumerate(self.workflow, start=1):
                lines.append(f"  {i}. {step}")

        # --- Available tools ---
        if self.tools:
            lines.append("")
            lines.append("## Available Tools")
            lines.append("You have access to the following tools to complete this skill:")
            for tool in self.tools:
                param_names = [p.name for p in tool.parameters]
                params_str = f"({', '.join(param_names)})" if param_names else "()"
                lines.append(f"  - {tool.name}{params_str}: {tool.description}")

        # --- Resources ---
        resources = self.resources()
        if resources:
            lines.append("")
            lines.append("## Resources")
            lines.append(
                "Paths in the instructions above are relative to the skill folder. Read these files with the "
                f"`read_skill_resource` tool (skill: `{self.name}`) when the instructions call for them:"
            )
            for resource in resources[:MAX_LISTED_RESOURCES]:
                lines.append(f"  - {resource}")
            if len(resources) > MAX_LISTED_RESOURCES:
                lines.append(f"  - ... and {len(resources) - MAX_LISTED_RESOURCES} more")

        # --- Examples ---
        if self.examples:
            lines.append("")
            lines.append("## Examples")
            for i, (user_msg, assistant_msg) in enumerate(self.examples, start=1):
                lines.append(f"  Example {i}:")
                lines.append(f"    User: {user_msg}")
                lines.append(f"    Assistant: {assistant_msg}")

        return "\n".join(lines)


def bundled_skills_dir() -> Path:
    """Folder holding the skills that ship with the package (currently ``raw-agents``)."""
    return Path(__file__).resolve().parent.parent / "bundled_skills"


def discover_skills(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    load_tools: bool = False,
) -> List[Skill]:
    """
    Load skills from one or more paths. A path is either a skill folder (it has a
    ``SKILL.md``) or a folder whose subfolders are skills. Duplicate names raise.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    skills: List[Skill] = []
    seen: Dict[str, Path] = {}
    for entry in paths:
        root = Path(entry)
        if not root.is_dir():
            raise ValueError(f"Skills path {root} is not a directory.")
        folders = [root] if (root / SKILL_FILE).is_file() else sorted(
            child for child in root.iterdir() if (child / SKILL_FILE).is_file()
        )
        for folder in folders:
            try:
                skill = Skill.from_directory(folder, load_tools=load_tools)
            except Exception as e:
                raise ValueError(f"Could not load skill from {folder}: {e}") from e
            if skill.name in seen:
                raise ValueError(f"Duplicate skill name '{skill.name}': {seen[skill.name]} and {folder}")
            seen[skill.name] = folder
            skills.append(skill)
    return skills


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            return str(json.loads(value))
        except ValueError:
            return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def _scalar(value: str) -> str:
    """A YAML scalar: quotes removed, or an unquoted value without its trailing ``# comment``."""
    value = value.strip()
    quoted = re.match(r"""("(?:[^"\\]|\\.)*"|'(?:[^']|'')*')""", value)
    if quoted:
        return _unquote(quoted.group(1))
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)] if str(value).strip() else []


def _first_paragraph(body: str) -> str:
    for paragraph in re.split(r"\n\s*\n", body.strip()):
        text = " ".join(line.strip() for line in paragraph.splitlines() if not line.lstrip().startswith("#"))
        if text:
            return text[:500]
    return ""


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse the YAML subset skills use: ``key: value``, inline ``[a, b]`` lists, ``- item``
    block lists, one level of nested ``key: value`` maps and ``>`` / ``|`` multi-line text.
    Lines it cannot make sense of are skipped, so unusual keys never block loading.
    A file without a leading ``---`` has no frontmatter. Returns ``(meta, body)``.
    """
    lines = text.lstrip("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, "\n".join(lines)
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ValueError(f"The frontmatter in {SKILL_FILE} is missing its closing '---'.")

    def indented(i: int) -> bool:
        return i < end and (not lines[i].strip() or lines[i][0] in " \t")

    meta: Dict[str, Any] = {}
    i = 1
    while i < end:
        line = lines[i]
        i += 1
        match = re.match(r"^([A-Za-z_][\w.-]*):(?:\s+(.*)|\s*)$", line)
        if not match:
            continue
        key, value = match.group(1), (match.group(2) or "").strip()
        if re.fullmatch(r"[>|][+-]?\d*", value):
            block: List[str] = []
            while indented(i):
                block.append(lines[i].strip())
                i += 1
            meta[key] = (" " if value.startswith(">") else "\n").join(block).strip()
        elif value == "" or value.startswith("#"):
            # A block list, a nested map, or nothing; lists may also sit at the key's own indentation.
            block = []
            while indented(i) or (i < end and lines[i].startswith("- ")):
                block.append(lines[i])
                i += 1
            content = [b for b in block if b.strip() and not b.lstrip().startswith("#")]
            if content and content[0].lstrip().startswith("- "):
                meta[key] = [_scalar(b.lstrip()[2:]) for b in content if b.lstrip().startswith("- ")]
            else:
                nested: Dict[str, Any] = {}
                for b in content:
                    m = re.match(r"^\s+([A-Za-z_][\w.-]*):(?:\s+(.*)|\s*)$", b)
                    if m:
                        nested[m.group(1)] = _scalar(m.group(2) or "")
                meta[key] = nested if nested else ""
        elif value.startswith("["):
            try:
                meta[key] = [str(v) for v in json.loads(value)]
            except ValueError:
                inner = value[1:-1] if value.endswith("]") else value[1:]
                meta[key] = [_unquote(v) for v in inner.split(",") if v.strip()]
        else:
            meta[key] = _scalar(value)
    return meta, "\n".join(lines[end + 1:])


def _load_tools_file(file: Path) -> List[Tool]:
    resolved = file.resolve()
    module_name = "_raw_skill_tools_" + hashlib.sha1(str(resolved).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    tools = getattr(module, "TOOLS", None)
    if not isinstance(tools, list) or not all(isinstance(t, Tool) for t in tools):
        raise ValueError(f"{file} must define TOOLS as a list of Tool objects.")
    return tools

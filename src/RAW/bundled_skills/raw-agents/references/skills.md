# Skills

RAW reads skills in the Claude `SKILL.md` format, so skills written for Claude work in a RAW `Agent` unchanged.

```
my-skills/
  meeting-notes/
    SKILL.md           # frontmatter + instructions
    references/*.md    # optional; read on demand
    tools.py           # optional, RAW only
```

```markdown
---
name: meeting-notes
description: Turn transcripts into minutes with decisions and action items. Use when the user pastes a transcript or asks "what did we decide".
---

# Meeting notes
1. Read the transcript ...
2. Use the layout in `references/template.md`.
```

## Loading

```python
agent = Agent(name="a", llm=llm, skills_dir="my-skills")             # folder of skills, one skill folder, or a list of either
agent = Agent(name="a", llm=llm, skills=[Skill(...)], skills_dir=...)  # both are combined; names must be unique

from RAW import Skill, discover_skills
skill = Skill.from_directory("my-skills/meeting-notes")             # one skill
skills = discover_skills(["my-skills", "/other/skills"])            # many; raises on duplicate names
```

Only `name` and `description` matter. `name` defaults to the folder name and `description` to the first paragraph, so a `SKILL.md` with no frontmatter still loads. Unknown keys (`license`, `allowed-tools`, `metadata`, `argument-hint`, ...) are kept in `skill.metadata` and ignored. RAW's own optional keys: `version`, `author`, `tags`, `triggers`, `exclusions`, `workflow` (lists can be `[a, b]` or `- item` lines). Only an unclosed `---` block is an error. Home-relative paths are not expanded: use `os.path.expanduser("~/.claude/skills")`.

You can also build a skill in code: `Skill(name=..., description=..., system_prompt=..., tools=[...], workflow=[...], examples=[("user", "assistant")])`.

## How the agent uses them (`skill_mode`)

- `"catalog"` (default): the system prompt lists `name: description` for each skill. The model calls the built-in **`load_skill(name)`** tool to read the full instructions, which also unlocks that skill's tools. Loaded skills stay loaded for the agent's lifetime. A skill's tools are not callable before it is loaded.
- `"inline"`: every skill's full prompt is in the system prompt and all skill tools are available from the start. Use it for a few small skills, or with a model that has no tool support (the agent switches automatically).
- Skills chosen by a `Router`, or passed as `agent(..., skill=skill)`, are always inline.

## Files and tools in a skill folder

- Other files in the folder are listed in the loaded skill's prompt; the model reads them with the built-in **`read_skill_resource(skill, path)`**. It refuses anything that resolves outside the skill folder (including `..` and symlinks), rejects binary files and truncates very long ones.
- RAW never runs `scripts/`. Wrap anything executable as a `Tool`.
- `tools.py` must define `TOOLS = [Tool(...), ...]`. It is imported only with `Agent(..., load_skill_tools=True)` or `Skill.from_directory(path, load_tools=True)` because importing runs its code. Enable that for skills you trust.
- Reserved tool names: `load_skill`, `read_skill_resource` (and `ask_clarification` inside sub-agents).

## The skill bundled with the package

`pip install raw-agents` ships a skill that teaches an AI coding assistant to use this library (the one you are reading). `RAW.bundled_skills_dir()` returns its folder, so a RAW agent can load it too: `Agent(..., skills_dir=RAW.bundled_skills_dir())`. For Claude Code use `raw-agents install-skill`.

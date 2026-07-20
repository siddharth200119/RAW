from pydantic import BaseModel
from typing import Optional, List
from .tools import Tool

class Skill(BaseModel):
    version: str = "1.0"
    author: Optional[str] = None
    name: str
    description: str
    tags: List[str] = []
    triggers: List[str] = []
    exclusions: List[str] = []
    system_prompt: str
    workflow: List[str]
    examples: List[tuple[str, str]]
    tools: List[Tool] = []

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
        lines.append(f"## Description")
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
            lines.append(f"## Tags")
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

        # --- Examples ---
        if self.examples:
            lines.append("")
            lines.append("## Examples")
            for i, (user_msg, assistant_msg) in enumerate(self.examples, start=1):
                lines.append(f"  Example {i}:")
                lines.append(f"    User: {user_msg}")
                lines.append(f"    Assistant: {assistant_msg}")

        return "\n".join(lines)
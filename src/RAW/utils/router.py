from typing import List, Dict, Any, Optional
from RAW.models import Tool, Message, jsonschema, Skill
from RAW.llms.base import BaseLLM
from RAW.utils import Logger
import json
import jsonschema as js_lib

_ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "selected_tools": {
            "type": "array",
            "items": {"type": "string"}
        },
        "selected_skills": {
            "type": "array",
            "items": {"type": "string"}
        },
        "conversation_summary": {"type": "string"}
    },
    "required": ["reasoning", "selected_tools", "selected_skills", "conversation_summary"]
}


class Router:
    """
    A Router class that takes an LLM on initialization and exposes a route() method.
    It evaluates the user message, conversation summary, user summary, and available tools/skills,
    and returns a selection of which tools and skills are appropriate for the context.

    If the LLM call fails or returns invalid output, it is retried `retries` times;
    after that the router falls back to every tool and skill (``"fallback": True``)
    so a routing failure never leaves the agent without tools.

    Give the router an LLM configured for deterministic output (temperature 0),
    separate from the agent's LLM if the agent samples at a higher temperature.
    """
    def __init__(
        self,
        llm: BaseLLM,
        logger: Optional[Logger] = None,
        retries: int = 1,
    ):
        self.llm = llm
        self.logger = logger or Logger(service_name="Router")
        self.retries = retries

        if not self.llm:
            raise ValueError("Router must be initialized with an LLM.")

    async def route(
        self,
        user_message: Message,
        conversation_summary: str,
        user_summary: str,
        available_tools: List[Tool],
        available_skills: List[Skill]
    ) -> Dict[str, Any]:
        """
        Evaluates the context and decides which tools and skills to route the task to.
        
        Returns a dict with 'selected_tools' (List of tool names), 'selected_skills'
        (List of skill names), 'reasoning', 'conversation_summary' (updated summary
        incorporating the current user message) and 'fallback' (True when routing
        failed and every tool and skill was selected).
        """
        user_text = user_message.content or ""
        self.logger.debug(f"Routing request for user message: {user_text[:50]}...")
        
        # Build tool descriptions
        tool_descriptions = []
        for tool in available_tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        tools_str = "\n".join(tool_descriptions) if tool_descriptions else "No tools available."

        # Build skill descriptions
        skill_descriptions = []
        for skill in available_skills:
            tags_str = ", ".join(skill.tags) if skill.tags else "none"
            triggers_str = ", ".join(skill.triggers) if skill.triggers else "none"
            exclusions_str = ", ".join(skill.exclusions) if skill.exclusions else "none"
            skill_descriptions.append(
                f"- {skill.name}: {skill.description}\n"
                f"  Tags: {tags_str}\n"
                f"  Triggers: {triggers_str}\n"
                f"  Exclusions: {exclusions_str}"
            )
        skills_str = "\n".join(skill_descriptions) if skill_descriptions else "No skills available."

        system_prompt = f"""
You are an intelligent Routing Agent. Your job is to analyze the current User Message and the context (Conversation Summary and User Summary), and decide which of the Available Tools and Available Skills are required to fulfill the user's request.

You must also produce an updated Conversation Summary that incorporates the current User Message into the prior Conversation Summary. Keep it concise and focused on the ongoing intent, decisions, and relevant context.

--- Available Tools ---
{tools_str}

--- Available Skills ---
{skills_str}

You must respond with a JSON object exactly matching this schema:
{{
    "reasoning": "A brief explanation of why you selected these tools and skills",
    "selected_tools": ["tool_name_1", "tool_name_2"],
    "selected_skills": ["skill_name_1", "skill_name_2"],
    "conversation_summary": "Updated concise summary of the conversation so far"
}}

Only select tools and skills that are strictly necessary to fulfill the request. If none are needed, return empty arrays.
"""

        prompt = f"""
{system_prompt}

Conversation Summary:
{conversation_summary}

User Summary:
{user_summary}

User Message:
{user_text}
"""

        schema = jsonschema(_ROUTER_SCHEMA)
        tool_names = [tool.name for tool in available_tools]
        skill_names = [skill.name for skill in available_skills]

        last_error: Optional[Exception] = None
        for attempt in range(1 + self.retries):
            try:
                # We use generate with a jsonschema to ensure structured JSON output
                response = await self.llm.generate(
                    prompt=prompt,
                    schema=schema,
                    stream=False
                )
                result = self._parse(response)
            except Exception as e:
                last_error = e
                self.logger.warning(f"Routing attempt {attempt + 1} failed: {e}")
                continue

            result["selected_tools"] = self._known(result["selected_tools"], tool_names, "tool")
            result["selected_skills"] = self._known(result["selected_skills"], skill_names, "skill")
            result["fallback"] = False
            self.logger.info(
                f"Router selection: {result['selected_tools']} tools, "
                f"{result['selected_skills']} skills"
            )
            return result

        self.logger.error(
            f"Routing failed after {1 + self.retries} attempts, using all tools and skills: {last_error}"
        )
        return {
            "reasoning": f"Routing failed: {last_error}",
            "selected_tools": tool_names,
            "selected_skills": skill_names,
            "conversation_summary": conversation_summary,
            "fallback": True,
        }

    @staticmethod
    def _parse(response: Any) -> Dict[str, Any]:
        # The response could be a string or a dict
        result = json.loads(response) if isinstance(response, str) else response
        js_lib.validate(result, _ROUTER_SCHEMA)
        return result

    def _known(self, names: List[str], allowed: List[str], kind: str) -> List[str]:
        unknown = [name for name in names if name not in allowed]
        if unknown:
            self.logger.warning(f"Router selected unknown {kind}(s), ignored: {unknown}")
        return [name for name in names if name in allowed]

from typing import List, Dict, Any, Optional
from RAW.models import Tool, Message, jsonschema, Skill
from RAW.llms.base import BaseLLM
from RAW.utils import Logger
import json

class Router:
    """
    A Router class that takes an LLM on initialization and exposes a route() method.
    It evaluates the user message, conversation summary, user summary, and available tools/skills,
    and returns a selection of which tools and skills are appropriate for the context.
    """
    def __init__(
        self,
        llm: BaseLLM,
        logger: Optional[Logger] = None,
    ):
        self.llm = llm
        self.logger = logger or Logger(service_name="Router")
        
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
        (List of skill names), 'reasoning', and 'conversation_summary' (updated summary
        incorporating the current user message).
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

        schema = jsonschema({
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
        })

        try:
            # We use generate with a jsonschema to ensure structured JSON output
            response = await self.llm.generate(
                prompt=prompt,
                schema=schema,
                stream=False
            )
            
            # The response could be a string or a dict
            if isinstance(response, str):
                result = json.loads(response)
            elif isinstance(response, dict):
                result = response
            else:
                self.logger.error("Router did not return valid content.")
                return {
                    "reasoning": "Error parsing LLM response",
                    "selected_tools": [],
                    "selected_skills": [],
                    "conversation_summary": conversation_summary,
                }

            self.logger.info(
                f"Router selection: {result.get('selected_tools', [])} tools, "
                f"{result.get('selected_skills', [])} skills"
            )
            return result
                
        except Exception as e:
            self.logger.error(f"Error during routing: {e}")
            return {
                "reasoning": str(e),
                "selected_tools": [],
                "selected_skills": [],
                "conversation_summary": conversation_summary,
            }

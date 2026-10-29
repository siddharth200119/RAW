from typing import TYPE_CHECKING, List, Dict, Any, Optional
from RAW.models import Tool, Message, jsonschema, Skill, NoulQuestion, NoulCriteria, NoulAnswer
from RAW.utils.logger import Logger
import json
import jsonschema as js_lib  # type: ignore[import-untyped]

if TYPE_CHECKING:
    # Type-only: importing RAW.llms pulls in the [agentic] extra.
    from RAW.llms.base import BaseLLM
    from RAW.deciders.base import BaseDecider

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
    A Router that takes either an LLM or a decision model and exposes a route() method.
    It evaluates the user message, conversation summary, user summary, and available tools/skills,
    and returns a selection of which tools and skills are appropriate for the context.

    With an LLM, one structured-output call selects tools/skills and updates the
    conversation summary. With a decider, each tool and skill gets a yes/no (Noul)
    question and is selected when its probability is at least ``decider_threshold``.
    Decision models return no text, so the conversation summary is passed through unchanged.

    If routing fails (the LLM or decider raises, or the LLM output is invalid) it is
    retried `retries` times; after that the router selects every tool and skill
    (``"fallback": True``) so a routing failure never leaves the agent without tools.
    Give the router an LLM configured for deterministic output (temperature 0).
    """
    def __init__(
        self,
        llm: Optional["BaseLLM"] = None,
        logger: Optional[Logger] = None,
        decider: Optional["BaseDecider"] = None,
        decider_threshold: float = 0.5,
        retries: int = 1,
    ):
        if (llm is None) == (decider is None):
            raise ValueError("Router must be initialized with exactly one of an LLM or a decider.")
        if not 0.0 <= decider_threshold <= 1.0:
            raise ValueError("decider_threshold must be between 0 and 1.")

        self.llm = llm
        self.decider = decider
        self.decider_threshold = decider_threshold
        self.retries = retries
        self.logger = logger or Logger(service_name="Router")

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
        incorporating the current user message; unchanged when routing with a decider).
        Decider routing also returns 'probabilities' ({'tools': {...}, 'skills': {...}}).
        'fallback' is True when routing failed and every tool and skill was selected.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1 + self.retries):
            try:
                if self.decider is not None:
                    result = await self._route_with_decider(
                        self.decider, user_message, conversation_summary, user_summary, available_tools, available_skills
                    )
                else:
                    assert self.llm is not None
                    result = await self._route_with_llm(
                        self.llm, user_message, conversation_summary, user_summary, available_tools, available_skills
                    )
            except Exception as e:
                last_error = e
                self.logger.warning(f"Routing attempt {attempt + 1} failed: {e}")
                continue
            result["fallback"] = False
            return result

        self.logger.error(
            f"Routing failed after {1 + self.retries} attempts, using all tools and skills: {last_error}"
        )
        return {
            "reasoning": f"Routing failed: {last_error}",
            "selected_tools": [t.name for t in available_tools],
            "selected_skills": [s.name for s in available_skills],
            "conversation_summary": conversation_summary,
            "fallback": True,
        }

    async def _route_with_decider(
        self,
        decider: "BaseDecider",
        user_message: Message,
        conversation_summary: str,
        user_summary: str,
        available_tools: List[Tool],
        available_skills: List[Skill]
    ) -> Dict[str, Any]:
        user_text = user_message.content or ""
        self.logger.debug(f"Routing request with decider for user message: {user_text[:50]}...")

        # Question names are positional so tool/skill names never need escaping.
        questions: Dict[str, NoulQuestion] = {}
        for i, tool in enumerate(available_tools):
            questions[f"tool_{i}"] = NoulQuestion(
                instructions=f"Is the tool `{tool.name}` needed to fulfil the user's message?",
                criteria=NoulCriteria(
                    true=f"The request needs what this tool does: {tool.description}",
                    false="The request can be fulfilled without this tool.",
                ),
            )
        for i, skill in enumerate(available_skills):
            true = f"The request matches this skill: {skill.description}"
            if skill.triggers:
                true += f" Triggers: {', '.join(skill.triggers)}."
            false = "The request is unrelated to this skill."
            if skill.exclusions:
                false += f" Also no when it involves: {', '.join(skill.exclusions)}."
            questions[f"skill_{i}"] = NoulQuestion(
                instructions=f"Should the skill `{skill.name}` be used for the user's message?",
                criteria=NoulCriteria(true=true, false=false),
            )

        result: Dict[str, Any] = {
            "reasoning": "",
            "selected_tools": [],
            "selected_skills": [],
            "conversation_summary": conversation_summary,
            "probabilities": {"tools": {}, "skills": {}},
        }
        if not questions:
            result["reasoning"] = "No tools or skills available."
            return result

        state = {
            "user_message": user_text,
            "conversation_summary": conversation_summary,
            "user_summary": user_summary,
        }
        # Errors propagate to route(), which retries and then falls back to everything.
        decision = await decider.decide(state=state, questions=questions)

        names = {"tools": [t.name for t in available_tools], "skills": [s.name for s in available_skills]}
        for kind, kind_names in names.items():
            for i, name in enumerate(kind_names):
                answer = decision.answers.get(f"{kind[:-1]}_{i}")
                if not isinstance(answer, NoulAnswer):
                    self.logger.warning(f"Decider returned no yes/no answer for {kind[:-1]} '{name}'.")
                    continue
                result["probabilities"][kind][name] = answer.noul
                if answer.noul >= self.decider_threshold:
                    result[f"selected_{kind}"].append(name)

        result["reasoning"] = (
            f"Selected by decider {decider.model} at threshold {self.decider_threshold}."
        )
        self.logger.info(
            f"Router selection: {result['selected_tools']} tools, "
            f"{result['selected_skills']} skills"
        )
        return result

    async def _route_with_llm(
        self,
        llm: "BaseLLM",
        user_message: Message,
        conversation_summary: str,
        user_summary: str,
        available_tools: List[Tool],
        available_skills: List[Skill]
    ) -> Dict[str, Any]:
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

        # Errors (including invalid output) propagate to route(), which retries
        # and then falls back to everything.
        response = await llm.generate(
            prompt=prompt,
            schema=schema,
            stream=False
        )
        # The response could be a string or a dict
        result = json.loads(response) if isinstance(response, str) else response
        if not isinstance(result, dict):
            raise ValueError(f"Router expected a JSON object, got {type(result).__name__}.")
        js_lib.validate(result, _ROUTER_SCHEMA)

        result["selected_tools"] = self._known(
            result["selected_tools"], [t.name for t in available_tools], "tool"
        )
        result["selected_skills"] = self._known(
            result["selected_skills"], [s.name for s in available_skills], "skill"
        )
        self.logger.info(
            f"Router selection: {result['selected_tools']} tools, "
            f"{result['selected_skills']} skills"
        )
        return result

    def _known(self, names: List[str], allowed: List[str], kind: str) -> List[str]:
        unknown = [name for name in names if name not in allowed]
        if unknown:
            self.logger.warning(f"Router selected unknown {kind}(s), ignored: {unknown}")
        return [name for name in names if name in allowed]

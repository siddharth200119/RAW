"""
Sub-agents: an :class:`~RAW.agent.Agent` exposed to another agent as a tool.

The parent calls the tool, waits for it like any other tool, and gets back only the
sub-agent's final answer; everything the sub-agent read or did stays in its own context.
When the sub-agent cannot continue without information, it asks through the built-in
``ask_clarification`` tool. The question comes back to the parent as a ``needs_input``
result, or goes straight to a human through ``on_question``.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional

from RAW.agent import Agent
from RAW.models import Tool, ToolParam
from RAW.utils.metrics import NullMetrics

ASK_TOOL = "ask_clarification"

QuestionHandler = Callable[[str, str], Awaitable[Optional[str]]]


@dataclass
class _Run:
    agent: Agent
    clarifications: int = 0
    question: str = ""


def agent_tool(
    name: str,
    description: str,
    factory: Callable[[], Agent],
    *,
    max_depth: int = 3,
    max_clarifications: int = 2,
    on_question: Optional[QuestionHandler] = None,
    question_timeout: Optional[float] = None,
    max_open_runs: int = 64,
) -> Tool:
    """
    Wrap ``factory`` (which builds a fresh sub-agent per delegated task) as a :class:`Tool`.

    Args:
        name: Tool name the parent calls.
        description: What the sub-agent is good at; shown to the parent's LLM.
        factory: Builds the sub-agent. Called once per task, so tasks never share history.
        max_depth: Longest allowed chain of nested sub-agents.
        max_clarifications: Questions one task may ask. After that the sub-agent is told to
            assume and finish.
        on_question: ``async (question, subagent_name) -> answer | None``. When given, questions
            go to it first (a human, a UI). ``None``, a timeout or an exception hands the
            question to the parent instead.
        question_timeout: Seconds to wait for ``on_question``.
        max_open_runs: Unanswered sub-agents kept for ``resume``; the oldest is dropped beyond it.
    """
    runs: Dict[str, _Run] = {}
    # One question at a time, so a terminal prompt never interleaves between parallel sub-agents.
    question_lock = asyncio.Lock()

    def make_ask_tool(run: _Run) -> Tool:
        def ask_clarification(question: str, agent: Optional[Agent] = None) -> str:
            if run.clarifications >= max_clarifications or agent is None:
                return "No more questions are allowed. Make your best assumption, state it, and finish the task."
            run.clarifications += 1
            run.question = question
            agent.halt({"needs_input": question})
            return "Your question was passed on. Stop now; the answer will arrive as the next message."

        return Tool(
            name=ASK_TOOL,
            description=(
                "Ask the agent that gave you this task a question, only when you cannot finish without "
                "the answer. For minor details make a reasonable assumption instead."
            ),
            parameters=[ToolParam(name="question", type="string", description="The question.", required=True)],
            function=ask_clarification,
        )

    async def ask_handler(question: str) -> Optional[str]:
        if on_question is None:
            return None
        async with question_lock:
            try:
                return await asyncio.wait_for(on_question(question, name), question_timeout)
            except Exception:
                return None

    async def delegate(
        task: str = "", resume: str = "", answer: str = "", agent: Optional[Agent] = None
    ) -> AsyncGenerator[Any, None]:
        parent = agent
        started = time.perf_counter()
        status = "error"

        def finish() -> None:
            if parent is not None:
                tags = {"agent": parent.name, "subagent": name, "status": status}
                parent._count("raw.subagent.calls", tags=tags)
                parent._observe("raw.subagent.duration_seconds", time.perf_counter() - started, tags=tags)

        try:
            if resume:
                run = runs.pop(resume, None)
                if run is None:
                    yield f"Error: there is no waiting sub-agent '{resume}'; it may have finished. Start a new task instead."
                    return
                if not answer:
                    runs[resume] = run
                    yield "Error: pass `answer` together with `resume`."
                    return
                ref, message = resume, answer
            else:
                if not task:
                    yield "Error: pass `task` to start a sub-agent."
                    return
                depth = getattr(parent, "_depth", 0) + 1
                if depth > max_depth:
                    yield f"Error: sub-agents may not be nested deeper than {max_depth} levels."
                    return
                sub = factory()
                sub._depth = depth
                if parent is not None and isinstance(sub.metrics, NullMetrics):
                    sub.metrics = parent.metrics
                run = _Run(sub)
                sub.control_tools = [*sub.control_tools, make_ask_tool(run)]
                ref, message = uuid.uuid4().hex[:8], task

            while True:
                result: Optional[str] = None
                failure: Optional[str] = None
                async for chunk in run.agent(message):
                    assert isinstance(chunk, dict)
                    content = chunk["content"]
                    if isinstance(content, str):
                        result = content
                    else:
                        if isinstance(content, dict) and content.get("error"):
                            failure = content.get("message") or content["error"]
                        yield {"subagent": name, "agent_ref": ref, "event": content}
                if parent is not None:
                    parent._absorb_usage(run.agent.last_run_usage)

                if run.agent.halted:
                    answered = await ask_handler(run.question)
                    if answered is not None:
                        message = answered
                        continue
                    runs[ref] = run
                    while len(runs) > max_open_runs:
                        runs.pop(next(iter(runs)))
                    status = "needs_input"
                    yield json.dumps({
                        "status": "needs_input",
                        "question": run.question,
                        "agent_ref": ref,
                        "how_to_answer": f"Call `{name}` again with resume='{ref}' and answer=<your answer>. "
                                         "Answer it yourself if you can; otherwise ask the user first.",
                    })
                    return

                if result is None:
                    yield f"Error: the sub-agent '{name}' stopped without an answer" + (f": {failure}" if failure else ".")
                    return
                status = "success"
                yield result
                return
        finally:
            finish()

    return Tool(
        name=name,
        description=(
            f"{description}\n\nHands a task to a sub-agent that works in its own context and returns only its "
            "final answer. Describe the task completely: it cannot see this conversation. If the result has "
            "status 'needs_input', answer the question yourself if you can, or ask the user, then call this "
            "tool again with `resume` and `answer`."
        ),
        parameters=[
            ToolParam(name="task", type="string", description="The complete task for the sub-agent."),
            ToolParam(name="resume", type="string", description="agent_ref of a sub-agent waiting for an answer."),
            ToolParam(name="answer", type="string", description="The answer to the sub-agent's question; used with resume."),
        ],
        function=delegate,
    )

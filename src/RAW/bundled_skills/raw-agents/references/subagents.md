# Sub-agents

`agent_tool` wraps an agent as a tool another agent can call. The sub-agent works in its own context (history, tools, skills, even LLM) and the parent receives only its final answer.

```python
from RAW import Agent, agent_tool

researcher = agent_tool(
    name="researcher",
    description="Looks things up in the knowledge base and reports the facts.",
    factory=lambda: Agent(name="researcher", llm=cheap_llm, tools=[search]),   # called once per task
    # optional:
    max_depth=3,            # nested sub-agents allowed
    max_clarifications=2,   # questions one task may ask
    on_question=None,       # async (question, subagent_name) -> answer | None
    question_timeout=None,  # seconds to wait for on_question
    max_open_runs=64,       # unanswered sub-agents kept for resume
)
boss = Agent(name="boss", llm=llm, tools=[researcher])
```

- The factory must return a **fresh** `Agent` each call; tasks never share history.
- The parent blocks on the call like on any tool. Several sub-agent calls in one reply run one after another.
- The tool takes `task`, `resume`, `answer`. The task text must be complete: the sub-agent cannot see the parent's conversation.
- Events inside the sub-agent stream to the caller as `{"tool_yield": {"subagent": name, "agent_ref": ref, "event": {...}}}`.
- Sub-agent token usage is added to the parent's `agent.usage`. Without its own `metrics` the sub-agent reports to the parent's.

## When the sub-agent has a question

Every sub-agent has a built-in `ask_clarification(question)` tool; calling it stops the sub-agent. Then:

1. With `on_question` set, the question goes to it first (terminal, UI), one at a time; the answer is fed back and the sub-agent continues.
2. If there is no handler, or it returns `None`, times out or raises, the parent receives `{"status": "needs_input", "question": ..., "agent_ref": ..., "how_to_answer": ...}` as the tool result. The parent answers from its own context, or asks the user, then calls the tool again with `resume="<agent_ref>", answer="..."`. The sub-agent continues with its history intact.

After `max_clarifications` the sub-agent is told to assume and finish. Waiting sub-agents live only as long as the tool object (nothing is persisted); resuming an unknown or finished `agent_ref` returns an error message.

## Errors

A sub-agent that hits its `max_iterations` or returns nothing yields `Error: the sub-agent '<name>' stopped without an answer: ...` as the tool result. Exceeding `max_depth` returns an error to the calling agent.

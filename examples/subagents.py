"""
Sub-agents: a parent agent delegates to a researcher and answers its questions.

The researcher has its own tool and context. When it needs information the task did not
include, it calls ``ask_clarification``; ``on_question`` forwards that to the terminal here.
Remove ``on_question`` and the parent receives the question as a ``needs_input`` result and
decides whether to answer it or ask you.

Needs a running vLLM server (see main.py).

    uv run python examples/subagents.py
"""
import asyncio

from RAW import Agent, InMemoryMetrics, Tool, agent_tool
from RAW.llms import VLLM
from RAW.models import ToolParam


def search_docs(query: str) -> str:
    return f"Docs say: the {query} limit is 100 requests per minute on the free plan."


async def ask_human(question: str, subagent: str) -> str:
    return await asyncio.to_thread(input, f"\n[{subagent} asks] {question}\n> ")


async def main() -> None:
    llm = VLLM(model="main-model", base_url="http://localhost:8000")
    metrics = InMemoryMetrics()

    search = Tool(
        name="search_docs",
        description="Search the product documentation.",
        parameters=[ToolParam(name="query", type="string", description="What to look up.", required=True)],
        function=search_docs,
    )

    researcher = agent_tool(
        name="researcher",
        description="Looks things up in the product documentation and reports the facts.",
        factory=lambda: Agent(name="researcher", llm=llm, tools=[search], base_prompt="Answer from the docs only."),
        on_question=ask_human,
        question_timeout=120,
    )

    boss = Agent(name="boss", llm=llm, tools=[researcher], metrics=metrics, max_iterations=10)

    async for chunk in boss("What is our API rate limit?"):
        content = chunk["content"]
        if isinstance(content, dict) and "tool_yield" in content:
            event = content["tool_yield"]
            print(f"  [{event['subagent']}] {event['event']}")
        elif isinstance(content, str):
            print(content)

    print("\nUsage incl. sub-agents:", boss.usage)
    print("Sub-agent calls:", metrics.counter("raw.subagent.calls"), "| LLM calls:", metrics.counter("raw.llm.calls"))


if __name__ == "__main__":
    asyncio.run(main())

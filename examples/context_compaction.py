"""
Context compaction with a secondary summarizer LLM.

Needs a running vLLM server for the main model (see main.py); the summaries are
written by a second, smaller model. A deliberately small ``context_window`` makes
compaction kick in after a few turns so you can watch it happen.

    uv run python examples/context_compaction.py
"""
import asyncio

from RAW import Agent
from RAW.llms import VLLM

MAIN_URL = "http://localhost:8000"
SUMMARIZER_URL = "http://localhost:8001"


async def main() -> None:
    llm = VLLM(base_url=MAIN_URL, model="main-model")
    summarizer = VLLM(base_url=SUMMARIZER_URL, model="small-model")

    agent = Agent(
        name="Chatty",
        llm=llm,
        compactor_llm=summarizer,  # leave out to let `llm` summarize its own history
        context_window=4000,       # tiny on purpose
        compact_trigger=0.70,
        compact_target=0.50,
    )

    for question in ["Tell me about Rome", "And about Athens", "Compare them", "What did I ask first?"]:
        async for chunk in agent(question):
            content = chunk["content"]
            if isinstance(content, dict) and "compaction" in content:
                print(f"\n[compacted: {content['compaction']}]")
            elif isinstance(content, str):
                print(content)

    print("\nSummary kept by the agent:\n", agent.context_summary)
    print("Total usage:", agent.usage)


if __name__ == "__main__":
    asyncio.run(main())

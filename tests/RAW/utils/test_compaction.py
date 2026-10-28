import sys
from typing import List

import pytest

from RAW.models import Message, ToolCall, Usage
from RAW.utils.compaction import (
    TRUNCATED_TOOL_RESULT,
    compact_messages,
    estimate_tokens,
    split_messages,
    summarize_messages,
)


@pytest.fixture(autouse=True)
def no_tiktoken(monkeypatch):
    # Importing None raises ImportError, so counts fall back to len(text) // 4.
    monkeypatch.setitem(sys.modules, "tiktoken", None)


class FakeSummarizer:
    def __init__(self, reply="SUMMARY", fail=False):
        self.reply = reply
        self.fail = fail
        self.prompts: List[str] = []

    async def chat(self, messages, schema=None, stream=False, tools=None):
        self.prompts.append(messages[0].content)
        if self.fail:
            raise RuntimeError("summarizer down")
        return Message(role="assistant", content=self.reply, usage=Usage(input_tokens=50, output_tokens=5))


def text(n_tokens: int, word="x") -> str:
    return word * (n_tokens * 4)


def user(n=10, word="u"):
    return Message(role="user", content=text(n, word))


def assistant(n=10, word="a"):
    return Message(role="assistant", content=text(n, word))


def call_pair(i, result_tokens=10):
    return [
        Message(role="assistant", tool_calls=[ToolCall(id=f"c{i}", name="t", arguments={"i": i})]),
        Message(role="tool", content=text(result_tokens, "r"), tool_call_id=f"c{i}"),
    ]


def system():
    return Message(role="system", content="sys")


def assert_pairs_intact(messages):
    for i, m in enumerate(messages):
        if m.role == "tool":
            assert messages[i - 1].role in ("assistant", "tool")
            j = i - 1
            while messages[j].role == "tool":
                j -= 1
            assert m.tool_call_id in [c.id for c in messages[j].tool_calls]


def test_estimate_counts_tool_calls_and_images():
    plain = estimate_tokens([Message(role="assistant", content="hi")])
    with_call = estimate_tokens(
        [Message(role="assistant", content="hi", tool_calls=[ToolCall(id="1", name="search", arguments={"q": "x" * 400})])]
    )
    assert with_call > plain + 90


async def test_small_history_is_untouched():
    msgs = [system(), user(), assistant()]
    result = await compact_messages(msgs, FakeSummarizer(), target_tokens=1000)
    assert result.method == "none"
    assert result.messages == msgs


async def test_truncates_old_tool_results_before_summarizing():
    old = call_pair(1, result_tokens=2000)
    msgs = [system(), user(), *old, assistant(), user(), *call_pair(2, result_tokens=800)]
    summarizer = FakeSummarizer()
    result = await compact_messages(msgs, summarizer, target_tokens=1000)

    assert result.method == "truncate"
    assert summarizer.prompts == []
    old_result = result.messages[3]
    assert old_result.tool_call_id == "c1"
    assert old_result.content == TRUNCATED_TOOL_RESULT.format(tokens=2000)
    # The latest turn keeps its full tool result.
    assert result.messages[-1].content == text(800, "r")
    assert result.tokens_after <= 1000 < result.tokens_before


async def test_summarizes_old_turns_and_keeps_recent_tail():
    msgs = [system()]
    for i in range(8):
        msgs += [user(100, f"q{i}"), assistant(100, f"a{i}")]
    summarizer = FakeSummarizer("THE SUMMARY")
    result = await compact_messages(msgs, summarizer, target_tokens=500, summary_tokens=50)

    assert result.method == "summarize"
    assert result.summary == "THE SUMMARY"
    assert result.tokens_after <= 500
    assert result.messages[0].role == "system"
    assert result.messages[1].role == "user"
    # The newest messages survive verbatim; the oldest are gone.
    assert result.messages[-1] == msgs[-1]
    assert msgs[1] not in result.messages
    assert result.summarized_messages == len(msgs) - len(result.messages)
    assert "q0" in summarizer.prompts[0]
    assert result.usage == Usage(input_tokens=50, output_tokens=5)


async def test_cut_never_separates_tool_call_from_result():
    msgs = [system(), user(50)]
    for i in range(10):
        msgs += call_pair(i, result_tokens=60)
    msgs.append(assistant(20))
    result = await compact_messages(msgs, FakeSummarizer(), target_tokens=400, summary_tokens=50)

    assert result.method == "summarize"
    assert_pairs_intact(result.messages)
    # Cut inside one long turn: the user message stays in front of the tail.
    assert result.messages[1].role == "user"
    assert result.messages[1] == msgs[1]
    assert result.messages[-1] == msgs[-1]
    assert len(result.messages) < len(msgs)


async def test_failed_summarizer_drops_old_messages_instead_of_raising():
    msgs = [system()]
    for i in range(8):
        msgs += [user(100), assistant(100)]
    result = await compact_messages(msgs, FakeSummarizer(fail=True), target_tokens=500, summary_tokens=50)

    assert result.method == "drop"
    assert result.summary == ""
    assert result.tokens_after <= 500
    assert len(result.messages) < len(msgs)


async def test_previous_summary_is_folded_in():
    msgs = [system()]
    for i in range(6):
        msgs += [user(100), assistant(100)]
    summarizer = FakeSummarizer("NEW")
    result = await compact_messages(
        msgs, summarizer, target_tokens=500, previous_summary="OLD FACTS", summary_tokens=50
    )

    assert "OLD FACTS" in summarizer.prompts[0]
    assert result.summary == "NEW"


async def test_large_input_is_summarized_in_chunks():
    msgs = [user(100), assistant(100)] * 4
    summarizer = FakeSummarizer("S")
    summary, usage = await summarize_messages(summarizer, msgs, chunk_tokens=250)

    assert len(summarizer.prompts) > 1
    assert "Summary of the conversation before this transcript:\nS" in summarizer.prompts[1]
    assert usage.input_tokens == 50 * len(summarizer.prompts)


async def test_empty_summary_counts_as_failure():
    with pytest.raises(ValueError):
        await summarize_messages(FakeSummarizer(""), [user()])


async def test_measured_tokens_override_a_low_local_estimate():
    msgs = [system()]
    for i in range(6):
        msgs += [user(50), assistant(50)]
    # Locally ~650 tokens fits a 1000 target, but the provider says the prompt is 4000.
    result = await compact_messages(
        msgs, FakeSummarizer(), target_tokens=1000, summary_tokens=50, measured_tokens=4000
    )

    assert result.method == "summarize"
    assert result.tokens_before == 4000
    assert result.tokens_after <= 1000


def test_split_returns_no_head_for_a_single_small_turn():
    head, tail = split_messages([user(10), assistant(10)], keep_tokens=5)
    assert head == []
    assert len(tail) == 2

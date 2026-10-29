"""
Context compaction: shrink a message history by truncating old tool results and
summarizing older turns.

This lives in the base tier, so it must not import ``RAW.llms`` at runtime; the
summarizer is any object with an async ``chat(messages=..., stream=False)``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from RAW.models import Message, Tool, Usage
from .tokens import count_tokens

if TYPE_CHECKING:
    from RAW.llms.base import BaseLLM

# Rough cost of one image and of per-message framing; providers differ.
IMAGE_TOKENS = 1000
MESSAGE_OVERHEAD_TOKENS = 4
TRUNCATED_TOOL_RESULT = "[tool result truncated to save context: {tokens} tokens]"

SUMMARY_PROMPT = """You are compacting the conversation history of an AI agent so it can keep working with a smaller context.

{previous}Write a summary of the transcript below that the agent can rely on instead of the original messages. Preserve:
- the user's goals and any constraints or preferences they stated
- decisions made and the reasons for them
- concrete facts the agent will need again: names, IDs, file paths, URLs, numbers, tool outputs that matter
- work that is finished, work still in progress, and open questions
Drop pleasantries and anything that no longer matters. Use plain prose or short bullet points, at most about {summary_tokens} tokens. Reply with the summary only.

<transcript>
{transcript}
</transcript>"""


@dataclass
class CompactionResult:
    messages: List[Message]
    summary: str
    tokens_before: int
    tokens_after: int
    # "none" (already small), "truncate" (tool results only), "summarize", "drop" (summarizer failed).
    method: str = "none"
    summarized_messages: int = 0
    usage: Usage = field(default_factory=Usage)


def estimate_tokens(
    messages: Sequence[Message],
    tools: Optional[Sequence[Tool]] = None,
    model: str = "",
) -> int:
    """Local token estimate covering text, tool calls, images and tool schemas."""
    total = 0
    for m in messages:
        total += MESSAGE_OVERHEAD_TOKENS
        total += count_tokens(m.content or "", model)
        if m.think_content:
            total += count_tokens(m.think_content, model)
        for call in m.tool_calls:
            total += count_tokens(f"{call.name} {json.dumps(call.arguments, default=str)}", model)
        total += IMAGE_TOKENS * len(m.images)
    for tool in tools or []:
        total += count_tokens(json.dumps(tool.to_dict(), default=str), model)
    return total


def truncate_tool_results(
    messages: Sequence[Message],
    protect_from: int,
    max_tokens: int = 500,
    model: str = "",
) -> List[Message]:
    """
    Replace tool results before index ``protect_from`` that exceed ``max_tokens``
    with a one-line placeholder. The ``tool_call_id`` is kept, so call/result
    pairs stay valid.
    """
    out: List[Message] = []
    for i, m in enumerate(messages):
        if i < protect_from and m.role == "tool" and m.content:
            tokens = count_tokens(m.content, model)
            if tokens > max_tokens:
                m = m.model_copy(update={"content": TRUNCATED_TOOL_RESULT.format(tokens=tokens)})
        out.append(m)
    return out


def _last_user_index(messages: Sequence[Message]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            return i
    return -1


def split_messages(
    messages: Sequence[Message],
    keep_tokens: int,
    model: str = "",
) -> Tuple[List[Message], List[Message]]:
    """
    Split ``messages`` (without the system message) into ``(head, tail)``: ``head``
    is summarized, ``tail`` is kept verbatim and costs about ``keep_tokens``.

    The cut never separates an assistant message with tool calls from its tool
    results, and the tail starts with a user message whenever possible because
    some providers reject a history that starts with an assistant turn. ``head``
    is empty when there is nothing older that can be summarized.
    """
    if not messages:
        return [], []

    start = len(messages)
    used = 0
    while start > 0:
        cost = estimate_tokens([messages[start - 1]], model=model)
        if used + cost > keep_tokens and start < len(messages):
            break
        used += cost
        start -= 1

    # Prefer a cut at a user message so the tail is a sequence of whole turns.
    turn_cut = start
    while turn_cut > 0 and messages[turn_cut].role != "user":
        turn_cut -= 1
    if turn_cut > 0 and estimate_tokens(messages[turn_cut:], model=model) <= keep_tokens:
        return list(messages[:turn_cut]), list(messages[turn_cut:])

    # The latest turn alone is too big: cut inside it. Go back to the assistant
    # message that issued the tool calls, keep the user message in front of the
    # tail and summarize everything in between.
    cut = start
    while cut > 0 and messages[cut].role == "tool":
        cut -= 1
    user_idx = _last_user_index(messages)
    if cut <= user_idx:
        # The cut is not inside the latest turn: fall back to the turn boundary.
        if turn_cut > 0:
            return list(messages[:turn_cut]), list(messages[turn_cut:])
        return [], list(messages)
    pinned = [messages[user_idx]] if user_idx >= 0 else []
    head = [m for i, m in enumerate(messages[:cut]) if i != user_idx]
    if not head:
        return [], list(messages)
    return head, pinned + list(messages[cut:])


def render_transcript(messages: Sequence[Message], max_chars_per_message: int = 4000) -> str:
    lines: List[str] = []
    for m in messages:
        text = m.content or ""
        if len(text) > max_chars_per_message:
            half = max_chars_per_message // 2
            text = f"{text[:half]}\n[... {len(text) - max_chars_per_message} characters omitted ...]\n{text[-half:]}"
        label = m.role if m.role != "tool" else f"tool result ({m.tool_call_id})"
        lines.append(f"[{label}] {text}".rstrip())
        for call in m.tool_calls:
            lines.append(f"[assistant called tool] {call.name}({json.dumps(call.arguments, default=str)[:max_chars_per_message]})")
        if m.images:
            lines.append(f"[{m.role} attached {len(m.images)} image(s)]")
    return "\n".join(lines)


def _chunk_by_tokens(messages: Sequence[Message], max_tokens: int, model: str) -> List[List[Message]]:
    chunks: List[List[Message]] = []
    current: List[Message] = []
    used = 0
    for m in messages:
        cost = estimate_tokens([m], model=model)
        if current and used + cost > max_tokens:
            chunks.append(current)
            current, used = [], 0
        current.append(m)
        used += cost
    if current:
        chunks.append(current)
    return chunks


async def summarize_messages(
    llm: "BaseLLM",
    messages: Sequence[Message],
    previous_summary: str = "",
    summary_tokens: int = 1000,
    chunk_tokens: int = 24000,
    model: str = "",
) -> Tuple[str, Usage]:
    """
    Summarize ``messages`` with ``llm``, folding in ``previous_summary``. Input larger
    than ``chunk_tokens`` is summarized in sequential chunks, each seeing the summary so far.
    """
    summary = previous_summary
    usage = Usage()
    for chunk in _chunk_by_tokens(messages, chunk_tokens, model):
        previous = f"Summary of the conversation before this transcript:\n{summary}\n\n" if summary else ""
        prompt = SUMMARY_PROMPT.format(
            previous=previous,
            summary_tokens=summary_tokens,
            transcript=render_transcript(chunk),
        )
        response = await llm.chat(messages=[Message(role="user", content=prompt)], schema=None, stream=False)
        assert isinstance(response, Message)
        if response.usage:
            usage = usage + response.usage
        text = (response.content or "").strip()
        if not text:
            raise ValueError("summarizer returned an empty summary")
        summary = text
    return summary, usage


async def compact_messages(
    messages: Sequence[Message],
    llm: "BaseLLM",
    target_tokens: int,
    previous_summary: str = "",
    tools: Optional[Sequence[Tool]] = None,
    summary_tokens: int = 1000,
    chunk_tokens: int = 24000,
    model: str = "",
    measured_tokens: Optional[int] = None,
) -> CompactionResult:
    """
    Bring ``messages`` (``messages[0]`` is the system message and is never touched)
    under ``target_tokens``.

    First truncates bulky tool results from before the latest user turn; if that is
    not enough, summarizes the oldest messages and keeps a recent tail. When the
    summarizer fails the oldest messages are dropped instead, so a failure here never
    stops the agent. The returned ``summary`` is the one to put in the system prompt.

    ``measured_tokens`` is the provider-reported size of ``messages``. When given, the
    local estimate is scaled to match it, so a low estimate cannot hide a full context.
    """
    system, body = list(messages[:1]), list(messages[1:])
    summary_cost = count_tokens(previous_summary, model) if previous_summary else 0
    fixed = estimate_tokens(system, tools, model) + summary_cost
    local_before = fixed + estimate_tokens(body, model=model)
    scale = measured_tokens / local_before if measured_tokens and local_before else 1.0
    before = int(local_before * scale)
    target = int(target_tokens / scale)

    if local_before <= target:
        return CompactionResult(list(messages), previous_summary, before, before)

    body = truncate_tool_results(body, protect_from=max(_last_user_index(body), 0), model=model)
    after_truncate = fixed + estimate_tokens(body, model=model)
    if after_truncate <= target:
        return CompactionResult(
            system + body, previous_summary, before, int(after_truncate * scale), method="truncate"
        )

    keep_tokens = max(target - fixed - int(summary_tokens / scale), 0)
    head, tail = split_messages(body, keep_tokens, model)
    if not head:
        # Nothing older to summarize: a single oversized turn. Report it as is.
        return CompactionResult(
            system + body, previous_summary, before, int(after_truncate * scale), method="truncate"
        )

    usage = Usage()
    try:
        summary, usage = await summarize_messages(
            llm, head, previous_summary, summary_tokens, chunk_tokens, model
        )
        method = "summarize"
    except Exception:
        summary, method = previous_summary, "drop"

    after = (
        estimate_tokens(system, tools, model)
        + (count_tokens(summary, model) if summary else 0)
        + estimate_tokens(tail, model=model)
    )
    return CompactionResult(
        system + tail, summary, before, int(after * scale),
        method=method, summarized_messages=len(head), usage=usage,
    )

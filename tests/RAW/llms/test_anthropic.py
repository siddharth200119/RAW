import json
import pytest
import httpx
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from RAW.llms.anthropic import AnthropicLLM, AnthropicOptions, prepare_output_schema, DEFAULT_MAX_TOKENS
from RAW.models import LLMCapability, Message, Image, Tool, ToolParam, ToolCall, Usage, jsonschema


def _response(payload, status_code=200):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = payload
    response.text = json.dumps(payload)
    return response


def _sse(*events) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events
    ).encode("utf-8")


def _stream(*chunks: bytes):
    async def gen():
        for chunk in chunks:
            yield chunk
    return gen()


def _tool(name="calc"):
    async def fn(x: int):
        return str(x)
    return Tool(
        name=name,
        description="calc tool",
        parameters=[ToolParam(name="x", type="integer", description="x", required=True)],
        function=fn
    )


WEB_SEARCH = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}

CITATION = {
    "type": "web_search_result_location",
    "url": "https://example.com/brand",
    "title": "Brand page",
    "encrypted_index": "EoUB...",
    "cited_text": "The brand was founded in 2020."
}

WEB_SEARCH_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Let me search.", "citations": None},
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "brand founded"}},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [{
                "type": "web_search_result",
                "url": "https://example.com/brand",
                "title": "Brand page",
                "encrypted_content": "EqgfCio...",
                "page_age": None
            }]
        },
        {"type": "text", "text": "It was founded in 2020.", "citations": [CITATION]},
        {"type": "tool_use", "id": "toolu_1", "name": "calc", "input": {"x": 2020, "nested": {"a": [1, 2]}}}
    ],
    "stop_reason": "tool_use",
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 4000,
        "cache_creation_input_tokens": 10,
        "server_tool_use": {"web_search_requests": 1}
    }
}


@pytest.fixture
def llm():
    return AnthropicLLM(api_key="test_key")


def test_init_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env_key")
    client = AnthropicLLM()
    assert client.model == "claude-sonnet-5-5"
    headers = client.client.async_client.headers
    assert headers["x-api-key"] == "env_key"
    assert headers["anthropic-version"] == "2026-10-21"
    assert "anthropic-beta" not in headers
    assert str(client.client.async_client.base_url).rstrip("/") == "https://api.anthropic.com/v1"
    for cap in (LLMCapability.COMPLETION, LLMCapability.TOOLS, LLMCapability.VISION, LLMCapability.STREAMING):
        assert cap in client.capabilities

    info = client.info()
    assert info.provider == "anthropic"
    assert info.max_tokens == DEFAULT_MAX_TOKENS
    assert info.context_window == 1000000


def test_custom_base_url_and_fallback_beta():
    client = AnthropicLLM(
        api_key="k",
        base_url="http://proxy.local/",
        options=AnthropicOptions(fallbacks="default"),
        betas=["some-beta"]
    )
    assert str(client.client.async_client.base_url).rstrip("/") == "http://proxy.local/v1"
    assert client.client.async_client.headers["anthropic-beta"] == "some-beta,server-side-fallback-2026-07-01"
    body = client._build_body([Message(role="user", content="hi")])
    assert body["fallbacks"] == "default"


def test_message_conversion(llm):
    image = Image.from_array(np.zeros((4, 4, 3), dtype=np.uint8))
    messages = [
        Message(role="system", content="You are a marketer."),
        Message(role="user", content="Look at this", images=[image]),
        Message(role="assistant", content="Calling tools", tool_calls=[
            ToolCall(id="toolu_a", name="calc", arguments={"x": 1}),
            ToolCall(id="toolu_b", name="calc", arguments={"x": 2}),
        ]),
        Message(role="tool", content="1", tool_call_id="toolu_a"),
        Message(role="tool", content="2", tool_call_id="toolu_b"),
        Message(role="user", content="Thanks"),
    ]
    body = llm._build_body(messages, tools=[_tool()])

    assert body["max_tokens"] == DEFAULT_MAX_TOKENS
    assert body["system"] == [{"type": "text", "text": "You are a marketer."}]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]

    user_blocks = body["messages"][0]["content"]
    assert user_blocks[0]["type"] == "image"
    assert user_blocks[0]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": image.to_base64()}
    assert user_blocks[1] == {"type": "text", "text": "Look at this"}

    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "Calling tools"},
        {"type": "tool_use", "id": "toolu_a", "name": "calc", "input": {"x": 1}},
        {"type": "tool_use", "id": "toolu_b", "name": "calc", "input": {"x": 2}},
    ]
    # Both tool results (and the following user text) share one user turn.
    assert body["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_a", "content": "1"},
        {"type": "tool_result", "tool_use_id": "toolu_b", "content": "2"},
        {"type": "text", "text": "Thanks"},
    ]
    assert body["tools"] == [{
        "name": "calc",
        "description": "calc tool",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "integer", "description": "x"}},
            "required": ["x"]
        }
    }]
    assert "stream" not in body
    assert "cache_control" not in body


def test_caching_flags():
    llm = AnthropicLLM(api_key="k", options=AnthropicOptions(cache_system_prompt=True, cache_ttl="1h", cache_conversation=True))
    messages = [
        Message(role="system", content="Brand Fact Sheet ..."),
        Message(role="user", content="Shared context", cache_breakpoint=True),
        Message(role="assistant", content="ok"),
        Message(role="user", content="Volatile question"),
    ]
    body = llm._build_body(messages)
    cache_control = {"type": "ephemeral", "ttl": "1h"}
    assert body["system"][-1]["cache_control"] == cache_control
    assert body["messages"][0]["content"][-1]["cache_control"] == cache_control
    assert "cache_control" not in body["messages"][2]["content"][-1]
    assert body["cache_control"] == cache_control


def test_caching_off_by_default(llm):
    body = llm._build_body([Message(role="system", content="s"), Message(role="user", content="u")])
    assert "cache_control" not in body["system"][0]
    assert "cache_control" not in body


def test_system_message_cache_breakpoint_without_option(llm):
    body = llm._build_body([Message(role="system", content="s", cache_breakpoint=True), Message(role="user", content="u")])
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_chat_direct_parses_server_tools_citations_and_usage():
    llm = AnthropicLLM(api_key="k", options=AnthropicOptions(native_tools=[WEB_SEARCH]))
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _response(WEB_SEARCH_RESPONSE)
        response = await llm.chat(messages=[Message(role="user", content="When was the brand founded?")], tools=[_tool()])

        body = mock_post.call_args[1]["json"]
        assert mock_post.call_args[0][0] == "/messages"
        assert body["tools"][0]["name"] == "calc"
        assert body["tools"][1] == WEB_SEARCH

    assert response.role == "assistant"
    assert response.content == "Let me search.It was founded in 2020."
    assert response.provider == "anthropic"
    assert response.provider_content == WEB_SEARCH_RESPONSE["content"]
    assert response.citations == [CITATION]
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "toolu_1"
    assert response.tool_calls[0].arguments == {"x": 2020, "nested": {"a": [1, 2]}}
    assert response.usage == Usage(input_tokens=100, output_tokens=50, cache_read_tokens=4000, cache_write_tokens=10)


@pytest.mark.asyncio
async def test_tool_round_trip_reuses_provider_content(llm):
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _response(WEB_SEARCH_RESPONSE)
        assistant = await llm.chat(messages=[Message(role="user", content="q")])

    tool_message = Message(role="tool", content="2020", tool_call_id="toolu_1")
    body = llm._build_body([Message(role="user", content="q"), assistant, tool_message])

    sent_assistant = body["messages"][1]
    assert sent_assistant["role"] == "assistant"
    # Original blocks re-sent (None-valued keys dropped), incl. encrypted search content.
    expected = [{k: v for k, v in b.items() if v is not None} for b in WEB_SEARCH_RESPONSE["content"]]
    assert sent_assistant["content"] == expected
    assert sent_assistant["content"][2]["content"][0]["encrypted_content"] == "EqgfCio..."
    assert body["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "2020"}]
    }
    # Re-serialising must not mutate the stored message.
    assert assistant.provider_content[0]["citations"] is None


def test_provider_content_ignored_for_other_providers(llm):
    message = Message(
        role="assistant",
        content="hello",
        provider="openai",
        provider_content=[{"type": "something_else"}]
    )
    body = llm._build_body([Message(role="user", content="hi"), message])
    assert body["messages"][1]["content"] == [{"type": "text", "text": "hello"}]


@pytest.mark.asyncio
async def test_chat_stream_assembles_tool_calls_and_usage(llm):
    events = _sse(
        {"type": "message_start", "message": {"id": "m", "role": "assistant", "content": [], "usage": {
            "input_tokens": 20, "output_tokens": 1, "cache_read_input_tokens": 300, "cache_creation_input_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Founded "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "in 2020."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "citations_delta", "citation": CITATION}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "calc", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"x\": 4"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "2, \"tags\": [\"a\"]}"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 30}},
        {"type": "message_stop"},
    )
    # Split mid-line to check SSE buffering across network chunks.
    chunks = [events[:137], events[137:401], events[401:]]

    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _stream(*chunks)
        generator = await llm.chat(messages=[Message(role="user", content="q")], tools=[_tool()], stream=True)
        received = [m async for m in generator]
        assert mock_post.call_args[1]["json"]["stream"] is True
        assert mock_post.call_args[1]["stream"] is True

    text = "".join(m.content for m in received if m.content)
    assert text == "Founded in 2020."
    final = received[-1]
    assert final.content is None
    assert final.tool_calls == [ToolCall(id="toolu_9", name="calc", arguments={"x": 42, "tags": ["a"]})]
    assert final.citations == [CITATION]
    assert final.usage == Usage(input_tokens=20, output_tokens=30, cache_read_tokens=300, cache_write_tokens=0)
    assert final.provider_content[0] == {"type": "text", "text": "Founded in 2020.", "citations": [CITATION]}
    assert final.provider_content[1]["input"] == {"x": 42, "tags": ["a"]}


@pytest.mark.asyncio
async def test_chat_stream_error_event_raises(llm):
    events = _sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _stream(events)
        generator = await llm.chat(messages=[Message(role="user", content="q")], stream=True)
        with pytest.raises(RuntimeError, match="overloaded_error"):
            [m async for m in generator]


@pytest.mark.asyncio
async def test_chat_stream_non_sse_error_body_raises(llm):
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _stream(b'{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}')
        generator = await llm.chat(messages=[Message(role="user", content="q")], stream=True)
        with pytest.raises(RuntimeError, match="invalid_request_error"):
            [m async for m in generator]


@pytest.mark.asyncio
async def test_pause_turn_is_resumed(llm):
    first = {
        "content": [{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}}],
        "stop_reason": "pause_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }
    second = {
        "content": [{"type": "text", "text": "Done."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 7}
    }
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [_response(first), _response(second)]
        response = await llm.chat(messages=[Message(role="user", content="q")])

        assert mock_post.call_count == 2
        resumed_messages = mock_post.call_args_list[1][1]["json"]["messages"]
        assert resumed_messages[-1] == {"role": "assistant", "content": first["content"]}
        assert len(resumed_messages) == 2

    assert response.content == "Done."
    assert response.provider_content == first["content"] + second["content"]
    assert response.usage == Usage(input_tokens=22, output_tokens=12)


@pytest.mark.asyncio
async def test_generate_uses_native_structured_output(llm):
    schema = jsonschema({
        "type": "object",
        "properties": {
            "headline": {"type": "string", "maxLength": 80},
            "tags": {"type": "array", "items": {"type": "object", "properties": {"minimum": {"type": "string"}}}}
        },
        "required": ["headline"]
    })
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _response({
            "content": [{"type": "text", "text": "{\"headline\": \"Hi\"}"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })
        output = await llm.generate(prompt="Write a headline", schema=schema)
        body = mock_post.call_args[1]["json"]

    assert output == "{\"headline\": \"Hi\"}"
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Write a headline"}]}]
    assert body["output_config"]["format"] == {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "tags": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"minimum": {"type": "string"}},
                    "additionalProperties": False
                }}
            },
            "required": ["headline"],
            "additionalProperties": False
        }
    }
    assert "tools" not in body


def test_prepare_output_schema_keeps_explicit_additional_properties():
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    assert prepare_output_schema(schema) == schema


@pytest.mark.asyncio
async def test_generate_stream_yields_text(llm):
    events = _sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 3, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    )
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _stream(events)
        generator = await llm.generate(prompt="hi", stream=True)
        assert [chunk async for chunk in generator] == ["Hello", " world"]


def test_effort_and_thinking_options():
    llm = AnthropicLLM(api_key="k", options=AnthropicOptions(effort="low", thinking={"type": "adaptive"}, max_tokens=256))
    body = llm._build_body([Message(role="user", content="hi")], schema={"type": "object", "properties": {}})
    assert body["max_tokens"] == 256
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["thinking"] == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_http_error_raises(llm):
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _response({"type": "error", "error": {"type": "invalid_request_error"}}, status_code=400)
        with pytest.raises(RuntimeError, match="400"):
            await llm.chat(messages=[Message(role="user", content="q")])


@pytest.mark.asyncio
async def test_count_tokens(llm):
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _response({"input_tokens": 7})
        tokens = await llm.count_tokens([Message(role="system", content="s"), Message(role="user", content="Hello")])
        assert tokens == 7
        assert mock_post.call_args[0][0] == "/messages/count_tokens"
        assert mock_post.call_args[1]["json"]["system"] == [{"type": "text", "text": "s"}]

    with patch.object(llm.client, "post", side_effect=RuntimeError("API down")):
        assert await llm.count_tokens("Hello world testing fallback") in [4, 5, 7]


@pytest.mark.asyncio
async def test_stop(llm):
    with patch.object(llm.client, "aclose", new_callable=AsyncMock) as mock_aclose:
        await llm.stop()
        mock_aclose.assert_called_once()


@pytest.mark.asyncio
async def test_stream_handles_utf8_split_across_chunks(llm):
    events = _sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Launch day 🚀"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
    )
    split = events.index("🚀".encode("utf-8")) + 2
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _stream(events[:split], events[split:])
        generator = await llm.chat(messages=[Message(role="user", content="q")], stream=True)
        received = [m async for m in generator]
    assert received[0].content == "Launch day 🚀"

import json
import pytest
import httpx
from unittest.mock import MagicMock, AsyncMock, patch
from RAW.llms.openai import OpenAILLM, OpenAIOptions, build_response_format
from RAW.llms.vllm import VLLM, VLLMOptions
from RAW.models import LLMCapability, Message, Tool, ToolCall, Usage, jsonschema


WEB_SEARCH = {"type": "web_search_preview"}
SCHEMA = jsonschema({"title": "Router Output", "type": "object", "properties": {"res": {"type": "string"}}})


def _vllm(options=None):
    def fake_init(self):
        self.capabilities = [LLMCapability.COMPLETION, LLMCapability.CHAT, LLMCapability.TOOLS]
    with patch.object(VLLM, "_initialize_capabilities", fake_init):
        return VLLM(base_url="http://127.0.0.1:8000", options=options)


@pytest.fixture(params=["openai", "vllm"])
def make_llm(request):
    def make(**option_kwargs):
        if request.param == "openai":
            return OpenAILLM(api_key="k", options=OpenAIOptions(**option_kwargs) if option_kwargs else None)
        return _vllm(VLLMOptions(**option_kwargs) if option_kwargs else None)
    return make


def _response(payload):
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = payload
    return response


def _tool():
    async def fn():
        return ""
    return Tool(name="calc", description="calc tool", parameters=[], function=fn)


def test_build_response_format():
    assert build_response_format(SCHEMA) == {
        "type": "json_schema",
        "json_schema": {"name": "Router_Output", "schema": dict(SCHEMA)}
    }
    assert build_response_format(SCHEMA, "json_object") == {"type": "json_object"}
    assert build_response_format(json.dumps(dict(SCHEMA)))["type"] == "json_schema"
    assert build_response_format("not json") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_native_tools_and_usage(make_llm):
    llm = make_llm(native_tools=[WEB_SEARCH], temperature=0.2)
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "prompt_tokens_details": {"cached_tokens": 100}}
    }
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _response(payload)
        response = await llm.chat(messages=[Message(role="user", content="q")], tools=[_tool()])
        body = mock_post.call_args[1]["json"]

    assert body["tools"][0]["function"]["name"] == "calc"
    assert body["tools"][1] == WEB_SEARCH
    # RAW-only options are not sent as request params.
    assert "native_tools" not in body
    assert "structured_output" not in body
    assert body["temperature"] == 0.2
    assert response.usage == Usage(input_tokens=20, output_tokens=30, cache_read_tokens=100)


@pytest.mark.asyncio
async def test_native_tools_per_call_without_raw_tools(make_llm):
    llm = make_llm()
    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _response({"choices": [{"message": {"content": "hi"}}]})
        response = await llm.chat(messages=[Message(role="user", content="q")], native_tools=[WEB_SEARCH])
        assert mock_post.call_args[1]["json"]["tools"] == [WEB_SEARCH]
    assert response.usage is None


@pytest.mark.asyncio
async def test_chat_json_schema_and_fallback(make_llm):
    for mode, expected in [
        ("json_schema", {"type": "json_schema", "json_schema": {"name": "Router_Output", "schema": dict(SCHEMA)}}),
        ("json_object", {"type": "json_object"}),
    ]:
        llm = make_llm(structured_output=mode)
        with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = _response({"choices": [{"message": {"content": "{}"}}]})
            await llm.chat(messages=[Message(role="user", content="q")], schema=SCHEMA)
            assert mock_post.call_args[1]["json"]["response_format"] == expected

            await llm.generate(prompt="q", schema=SCHEMA)
            assert mock_post.call_args[1]["json"]["response_format"] == expected


@pytest.mark.asyncio
async def test_stream_tool_calls_and_usage_across_split_chunks(make_llm):
    llm = make_llm()
    events = [
        {"choices": [{"delta": {"content": "Let me "}}]},
        {"choices": [{"delta": {"content": "check."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "calc", "arguments": "{\"x\": "}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"y\": [1, 2]}}"}}]}}]},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 9}},
    ]
    raw = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    raw = raw.encode("utf-8")
    chunks = [raw[i:i + 37] for i in range(0, len(raw), 37)]

    async def gen():
        for chunk in chunks:
            yield chunk

    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = gen()
        generator = await llm.chat(messages=[Message(role="user", content="q")], tools=[_tool()], stream=True)
        received = [m async for m in generator]
        assert mock_post.call_args[1]["json"]["stream_options"] == {"include_usage": True}

    assert "".join(m.content for m in received if m.content) == "Let me check."
    final = received[-1]
    assert final.tool_calls == [ToolCall(id="call_1", name="calc", arguments={"x": {"y": [1, 2]}})]
    assert final.usage == Usage(input_tokens=50, output_tokens=9)


@pytest.mark.asyncio
async def test_stream_usage_only_final_message(make_llm):
    llm = make_llm()
    raw = (
        f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}\n\n"
        f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 5, 'completion_tokens': 1}})}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")

    async def gen():
        yield raw

    with patch.object(llm.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = gen()
        generator = await llm.chat(messages=[Message(role="user", content="q")], stream=True)
        received = [m async for m in generator]

    assert [m.content for m in received] == ["hi", None]
    assert received[-1].tool_calls == []
    assert received[-1].usage == Usage(input_tokens=5, output_tokens=1)

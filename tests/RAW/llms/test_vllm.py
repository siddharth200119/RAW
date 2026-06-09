import pytest
import json
import httpx
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from RAW.llms.vllm import VLLM, VLLMOptions
from RAW.modals import LLMCapability, Message, LLMInfo, jsonschema, Tool, ToolCall, Image


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def vllm_client(mock_logger):
    options = VLLMOptions(temperature=0.7, max_tokens=100)
    client = VLLM(
        model="Qwen/Qwen2.5-14B-Instruct-AWQ",
        base_url="http://127.0.0.1:8000",
        options=options,
        logger=mock_logger
    )
    return client


def test_vllm_init(vllm_client):
    assert vllm_client.model == "Qwen/Qwen2.5-14B-Instruct-AWQ"
    assert vllm_client.options.temperature == 0.7
    assert LLMCapability.TOOLS in vllm_client.capabilities
    assert LLMCapability.COMPLETION in vllm_client.capabilities


@pytest.mark.asyncio
async def test_vllm_info(vllm_client):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200

    with patch.object(vllm_client.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        info = await vllm_client.info()
        assert isinstance(info, LLMInfo)
        assert info.model_name == "Qwen/Qwen2.5-14B-Instruct-AWQ"
        assert info.provider == "vllm"
        assert info.max_tokens == 100
        assert info.context_window == 32768
        assert LLMCapability.TOOLS in info.capabilities


@pytest.mark.asyncio
async def test_vllm_count_tokens_tokenize_success(vllm_client):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    # Simulate a list of 5 tokens
    mock_response.json.return_value = {"tokens": [1, 2, 3, 4, 5]}

    with patch.object(vllm_client.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        tokens = await vllm_client.count_tokens("Hello world")
        assert tokens == 5
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8000/tokenize",
            json={"model": "Qwen/Qwen2.5-14B-Instruct-AWQ", "prompt": "Hello world"},
            is_async=True
        )


@pytest.mark.asyncio
async def test_vllm_count_tokens_fallback(vllm_client):
    # Simulate a post failure
    with patch.object(vllm_client.client, "post", side_effect=RuntimeError("API down")):
        tokens = await vllm_client.count_tokens("Hello world testing fallback")
        # "Hello world testing fallback" is 28 chars. 28 // 4 = 7
        assert tokens == 7


@pytest.mark.asyncio
async def test_vllm_generate_direct(vllm_client):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "output text"
            }
        }]
    }

    with patch.object(vllm_client.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        schema = jsonschema({"type": "object", "properties": {"res": {"type": "string"}}})
        output = await vllm_client.generate(prompt="Hello", schema=schema)
        assert output == "output text"
        
        # Verify JSON format parameters passed in post body
        call_args = mock_post.call_args[1]
        assert call_args["json"]["response_format"] == {"type": "json_object"}
        assert "json schema" in call_args["json"]["messages"][0]["content"][0]["text"]


@pytest.mark.asyncio
async def test_vllm_chat_direct(vllm_client):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "response content",
                "tool_calls": [{
                    "id": "c1",
                    "function": {
                        "name": "calc",
                        "arguments": '{"x": 10}'
                    }
                }]
            }
        }]
    }

    with patch.object(vllm_client.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="compute")]
        
        async def dummy_fn():
            pass
            
        tool = Tool(name="calc", description="calc tool", parameters=[], function=dummy_fn)
        
        response = await vllm_client.chat(messages=messages, tools=[tool])
        assert isinstance(response, Message)
        assert response.role == "assistant"
        assert response.content == "response content"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "calc"
        assert response.tool_calls[0].arguments == {"x": 10}


@pytest.mark.asyncio
async def test_vllm_embed(vllm_client):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}

    with patch.object(vllm_client.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        embedding = await vllm_client.embed("hello")
        assert np.array_equal(embedding, np.array([0.1, 0.2, 0.3], dtype=np.float32))


@pytest.mark.asyncio
async def test_vllm_stop(vllm_client):
    with patch.object(vllm_client.client, "aclose", new_callable=AsyncMock) as mock_aclose:
        await vllm_client.stop()
        mock_aclose.assert_called_once()

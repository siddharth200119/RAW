import pytest
import httpx
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from RAW.llms.vllm import VLLM, VLLMOptions
from RAW.llms.openai import OpenAILLM, OpenAIOptions
from RAW.llms.gemini import GeminiLLM, GeminiOptions
from RAW.models import LLMCapability, Message, LLMInfo, jsonschema, Tool, ToolCall


@pytest.fixture(params=[
    (VLLM, VLLMOptions, "Qwen/Qwen2.5-14B-Instruct-AWQ", "vllm", "http://127.0.0.1:8000"),
    (OpenAILLM, OpenAIOptions, "gpt-4o-mini", "openai", "https://api.openai.com"),
    (GeminiLLM, GeminiOptions, "gemini-2.5-flash-lite", "gemini", "https://generativelanguage.googleapis.com/v1beta")
])
def llm_provider_setup(request):
    llm_class, options_class, model_name, provider_name, default_base_url = request.param
    if options_class is GeminiOptions:
        options = options_class(temperature=0.7, max_output_tokens=100)
    else:
        options = options_class(temperature=0.7, max_tokens=100)
    
    # Initialize LLM client
    if llm_class is OpenAILLM:
        client = llm_class(model=model_name, base_url=default_base_url, options=options, api_key="test_key")
    elif llm_class is GeminiLLM:
        client = llm_class(model=model_name, options=options, api_key="test_key")
    else:
        client = llm_class(model=model_name, base_url=default_base_url, options=options)
        
    return client, model_name, provider_name, default_base_url


def test_llm_init(llm_provider_setup):
    client, model_name, provider_name, _ = llm_provider_setup
    assert client.model == model_name
    assert client.options.temperature == 0.7
    assert LLMCapability.COMPLETION in client.capabilities


@pytest.mark.asyncio
async def test_llm_info(llm_provider_setup):
    client, model_name, provider_name, _ = llm_provider_setup
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200

    with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        info = await client.info()
        assert isinstance(info, LLMInfo)
        assert info.model_name == model_name
        assert info.provider == provider_name
        assert info.max_tokens == 100
        assert info.context_window in [16384, 32768, 1048576]
        assert LLMCapability.TOOLS in info.capabilities


@pytest.mark.asyncio
async def test_llm_count_tokens_tokenize_success(llm_provider_setup):
    client, model_name, provider_name, default_base_url = llm_provider_setup
    
    if provider_name == "openai":
        tokens = await client.count_tokens("Hello world")
        # Fallback estimation or tiktoken (approx 4 chars per token fallback since tiktoken might not be present in test env)
        assert tokens in [2, 3] # "Hello world" is 11 chars. 11 // 4 = 2. tiktoken cl100k_base: ["Hello", " world"] is 2 tokens.
    elif provider_name == "gemini":
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"totalTokens": 5}

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            tokens = await client.count_tokens("Hello world")
            assert tokens == 5
            mock_post.assert_called_once_with(
                f"/models/{model_name}:countTokens?key=test_key",
                json={"contents": [{"parts": [{"text": "Hello world"}]}]},
                is_async=True
            )
    else:
        # VLLM makes a POST request to root url /tokenize
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"tokens": [1, 2, 3, 4, 5]}

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            tokens = await client.count_tokens("Hello world")
            assert tokens == 5
            mock_post.assert_called_once_with(
                f"{default_base_url}/tokenize",
                json={"model": model_name, "prompt": "Hello world"},
                is_async=True
            )


@pytest.mark.asyncio
async def test_llm_count_tokens_fallback(llm_provider_setup):
    client, model_name, provider_name, _ = llm_provider_setup
    if provider_name == "openai":
        # OpenAI uses utils count_tokens directly, which never throws unless imports or types are broken
        tokens = await client.count_tokens("Hello world testing fallback")
        assert tokens in [4, 5, 7] # "Hello world testing fallback" is 28 chars. 28 // 4 = 7. tiktoken cl100k: 5 tokens.
    elif provider_name == "gemini":
        with patch.object(client.client, "post", side_effect=RuntimeError("API down")):
            tokens = await client.count_tokens("Hello world testing fallback")
            assert tokens in [4, 5, 7]
    else:
        # VLLM count_tokens falls back to utils count_tokens on POST error
        with patch.object(client.client, "post", side_effect=RuntimeError("API down")):
            tokens = await client.count_tokens("Hello world testing fallback")
            assert tokens in [4, 5, 7] # 28 chars: fallback 28 // 4 = 7, cl100k: 5 tokens


@pytest.mark.asyncio
async def test_llm_generate_direct(llm_provider_setup):
    client, model_name, provider_name, _ = llm_provider_setup
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    if provider_name == "gemini":
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "output text"}]
                }
            }]
        }
    else:
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "output text"
                }
            }]
        }

    with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        schema = jsonschema({"type": "object", "properties": {"res": {"type": "string"}}})
        output = await client.generate(prompt="Hello", schema=schema)
        assert output == "output text"
        
        call_args = mock_post.call_args[1]
        if provider_name == "gemini":
            assert call_args["json"]["generationConfig"]["responseMimeType"] == "application/json"
            assert call_args["json"]["generationConfig"]["responseSchema"] == dict(schema)
        else:
            assert call_args["json"]["response_format"] == {"type": "json_object"}
            assert "json schema" in call_args["json"]["messages"][0]["content"][0]["text"]


@pytest.mark.asyncio
async def test_llm_chat_direct(llm_provider_setup):
    client, model_name, provider_name, _ = llm_provider_setup
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    if provider_name == "gemini":
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "response content"},
                        {"functionCall": {"name": "calc", "args": {"x": 10}}}
                    ]
                }
            }]
        }
    else:
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

    with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        messages = [Message(role="user", content="compute")]
        
        async def dummy_fn():
            pass
            
        tool = Tool(name="calc", description="calc tool", parameters=[], function=dummy_fn)
        
        response = await client.chat(messages=messages, tools=[tool])
        assert isinstance(response, Message)
        assert response.role == "assistant"
        assert response.content == "response content"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "calc"
        assert response.tool_calls[0].arguments == {"x": 10}


@pytest.mark.asyncio
async def test_llm_embed(llm_provider_setup):
    client, model_name, provider_name, _ = llm_provider_setup
    if provider_name == "gemini":
        with pytest.raises(NotImplementedError):
            await client.embed("hello")
    else:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            embedding = await client.embed("hello")
            assert np.array_equal(embedding, np.array([0.1, 0.2, 0.3], dtype=np.float32))


@pytest.mark.asyncio
async def test_llm_stop(llm_provider_setup):
    client, model_name, provider_name, _ = llm_provider_setup
    with patch.object(client.client, "aclose", new_callable=AsyncMock) as mock_aclose:
        await client.stop()
        mock_aclose.assert_called_once()


def test_vllm_dynamic_capabilities_success():
    def mock_post(url, json, timeout=None):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        return mock_resp

    with patch("RAW.utils.requests.RequestsClient.post", side_effect=mock_post) as mock_post_method:
        client = VLLM(model="Qwen/Qwen2.5-14B-Instruct-AWQ", base_url="http://127.0.0.1:11434")
        
        assert LLMCapability.COMPLETION in client.capabilities
        assert LLMCapability.CHAT in client.capabilities
        assert LLMCapability.STREAMING in client.capabilities
        assert LLMCapability.TOOLS in client.capabilities
        assert LLMCapability.VISION in client.capabilities
        assert LLMCapability.EMBEDDING in client.capabilities
        assert mock_post_method.call_count == 4


def test_vllm_dynamic_capabilities_partial():
    def mock_post(url, json, timeout=None):
        mock_resp = MagicMock(spec=httpx.Response)
        content = json.get("messages", [{}])[0].get("content")
        has_image = False
        if isinstance(content, list):
            has_image = any(item.get("type") == "image_url" for item in content if isinstance(item, dict))

        if url == "/chat/completions" and "tools" not in json and not has_image:
            mock_resp.status_code = 200
        else:
            mock_resp.status_code = 400
        return mock_resp

    with patch("RAW.utils.requests.RequestsClient.post", side_effect=mock_post):
        client = VLLM(model="Qwen/Qwen2.5-14B-Instruct-AWQ", base_url="http://127.0.0.1:11434")
        
        assert LLMCapability.COMPLETION in client.capabilities
        assert LLMCapability.CHAT in client.capabilities
        assert LLMCapability.STREAMING in client.capabilities
        assert LLMCapability.TOOLS not in client.capabilities
        assert LLMCapability.VISION not in client.capabilities
        assert LLMCapability.EMBEDDING not in client.capabilities


def test_vllm_dynamic_capabilities_connection_error():
    with patch("RAW.utils.requests.RequestsClient.post", side_effect=httpx.ConnectError("Connection refused")):
        client = VLLM(model="Qwen/Qwen2.5-14B-Instruct-AWQ", base_url="http://127.0.0.1:11434")
        
        fallback_expected = [LLMCapability.TOOLS, LLMCapability.COMPLETION]
        assert client.capabilities == fallback_expected


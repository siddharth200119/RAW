"""
Tests for GeminiLLM class
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from RAW.llms.gemini import GeminiLLM, GeminiOptions
from RAW.modals import Message, Image, Tool, ToolCall, LLMCapability
from RAW.modals.tools import ToolParam


class TestGeminiOptions:
    """Tests for GeminiOptions model"""
    
    def test_default_options(self):
        """Test that default options are None"""
        options = GeminiOptions()
        assert options.temperature is None
        assert options.top_p is None
        assert options.top_k is None
        assert options.max_output_tokens is None
        assert options.stop_sequences is None
    
    def test_custom_options(self):
        """Test custom options values"""
        options = GeminiOptions(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=1024,
            stop_sequences=["END", "STOP"]
        )
        assert options.temperature == 0.7
        assert options.top_p == 0.9
        assert options.top_k == 40
        assert options.max_output_tokens == 1024
        assert options.stop_sequences == ["END", "STOP"]


class TestGeminiLLMInit:
    """Tests for GeminiLLM initialization"""
    
    def test_default_init(self):
        """Test default initialization"""
        llm = GeminiLLM(api_key="test-key")
        assert llm.api_key == "test-key"
        # Default model is now gemini-3-flash-preview
        assert llm.model == "gemini-3-flash-preview"
        assert llm.options is None
        assert LLMCapability.COMPLETION in llm.capabilities
        assert LLMCapability.TOOLS in llm.capabilities
    
    def test_custom_model(self):
        """Test initialization with custom model"""
        llm = GeminiLLM(api_key="test-key", model="gemini-1.5-pro")
        assert llm.model == "gemini-1.5-pro"
    
    def test_with_options(self):
        """Test initialization with options"""
        options = GeminiOptions(temperature=0.5)
        llm = GeminiLLM(api_key="test-key", options=options)
        assert llm.options.temperature == 0.5


class TestBuildGenerationConfig:
    """Tests for _build_generation_config method"""
    
    def test_empty_config_without_options(self):
        """Test empty config when no options provided"""
        llm = GeminiLLM(api_key="test-key")
        config = llm._build_generation_config()
        assert config == {}
    
    def test_config_with_all_options(self):
        """Test config with all options set"""
        options = GeminiOptions(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=1024,
            stop_sequences=["END"]
        )
        llm = GeminiLLM(api_key="test-key", options=options)
        config = llm._build_generation_config()
        
        assert config["temperature"] == 0.7
        assert config["topP"] == 0.9
        assert config["topK"] == 40
        assert config["maxOutputTokens"] == 1024
        assert config["stopSequences"] == ["END"]
    
    def test_config_with_partial_options(self):
        """Test config with only some options set"""
        options = GeminiOptions(temperature=0.5, max_output_tokens=512)
        llm = GeminiLLM(api_key="test-key", options=options)
        config = llm._build_generation_config()
        
        assert config["temperature"] == 0.5
        assert config["maxOutputTokens"] == 512
        assert "topP" not in config
        assert "topK" not in config


class TestConvertMessageToGeminiFormat:
    """Tests for _convert_message_to_gemini_format method"""
    
    def test_user_message(self):
        """Test converting user message"""
        llm = GeminiLLM(api_key="test-key")
        message = Message(role="user", content="Hello")
        
        result = llm._convert_message_to_gemini_format(message)
        
        assert result["role"] == "user"
        assert result["parts"] == [{"text": "Hello"}]
    
    def test_assistant_message(self):
        """Test converting assistant message"""
        llm = GeminiLLM(api_key="test-key")
        message = Message(role="assistant", content="Hi there!")
        
        result = llm._convert_message_to_gemini_format(message)
        
        assert result["role"] == "model"
        assert result["parts"] == [{"text": "Hi there!"}]
    
    def test_system_message(self):
        """Test converting system message (mapped to user)"""
        llm = GeminiLLM(api_key="test-key")
        message = Message(role="system", content="You are helpful")
        
        result = llm._convert_message_to_gemini_format(message)
        
        assert result["role"] == "user"
        assert result["parts"] == [{"text": "You are helpful"}]
    
    def test_message_with_tool_calls(self):
        """Test converting message with tool calls"""
        llm = GeminiLLM(api_key="test-key")
        tool_call = ToolCall(name="get_weather", arguments={"city": "NYC"})
        message = Message(role="assistant", content="", tool_calls=[tool_call])
        
        result = llm._convert_message_to_gemini_format(message)
        
        assert result["role"] == "model"
        assert len(result["parts"]) == 1
        assert result["parts"][0]["functionCall"]["name"] == "get_weather"
        assert result["parts"][0]["functionCall"]["args"] == {"city": "NYC"}
    
    def test_tool_response_message(self):
        """Test converting tool response message"""
        llm = GeminiLLM(api_key="test-key")
        
        # Previous assistant message with tool call
        tool_call = ToolCall(name="get_weather", arguments={"city": "NYC"})
        assistant_msg = Message(role="assistant", content="", tool_calls=[tool_call])
        
        # Tool response
        tool_msg = Message(role="tool", content='{"temp": 72}')
        
        result = llm._convert_message_to_gemini_format(tool_msg, prev_messages=[assistant_msg])
        
        assert result["role"] == "function"
        assert result["parts"][0]["functionResponse"]["name"] == "get_weather"
        assert result["parts"][0]["functionResponse"]["response"]["content"] == '{"temp": 72}'


class TestConvertToolToGeminiFormat:
    """Tests for _convert_tool_to_gemini_format method"""
    
    def test_basic_tool_conversion(self):
        """Test converting a basic tool"""
        llm = GeminiLLM(api_key="test-key")
        
        tool = Tool(
            name="get_weather",
            description="Get weather for a city",
            parameters=[
                ToolParam(
                    name="city",
                    type="string",
                    description="City name",
                    required=True
                )
            ],
            function=lambda city: f"Weather in {city}"
        )
        
        result = llm._convert_tool_to_gemini_format(tool)
        
        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather for a city"
        assert result["parameters"]["type"] == "object"
        assert "city" in result["parameters"]["properties"]
        assert result["parameters"]["properties"]["city"]["type"] == "string"
        assert "city" in result["parameters"]["required"]
    
    def test_tool_with_optional_params(self):
        """Test converting tool with optional parameters"""
        llm = GeminiLLM(api_key="test-key")
        
        tool = Tool(
            name="search",
            description="Search for items",
            parameters=[
                ToolParam(name="query", type="string", description="Search query", required=True),
                ToolParam(name="limit", type="integer", description="Max results", required=False)
            ],
            function=lambda query, limit=10: f"Searching {query}"
        )
        
        result = llm._convert_tool_to_gemini_format(tool)
        
        assert "query" in result["parameters"]["required"]
        assert "limit" not in result["parameters"]["required"]
    
    def test_tool_name_sanitization(self):
        """Test that invalid tool names are sanitized"""
        llm = GeminiLLM(api_key="test-key")
        
        tool = Tool(
            name="invalid name!@#",
            description="Test tool",
            parameters=[],
            function=lambda: "test"
        )
        
        result = llm._convert_tool_to_gemini_format(tool)
        
        # Should be sanitized to remove invalid characters
        assert "!" not in result["name"]
        assert "@" not in result["name"]
        assert "#" not in result["name"]
        assert " " not in result["name"]


class TestGenerateMethod:
    """Tests for generate method"""
    
    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic generation"""
        llm = GeminiLLM(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello! How can I help?"}]
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(llm.client, 'request_async', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await llm.generate("Hello")
            
            assert result == "Hello! How can I help?"
            mock_request.assert_called_once()
            
            # Verify URL
            call_args = mock_request.call_args
            assert "/models/gemini-3-flash-preview:generateContent" in call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_generate_with_schema(self):
        """Test generation with JSON schema"""
        llm = GeminiLLM(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": '{"name": "test"}'}]
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(llm.client, 'request_async', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            result = await llm.generate("Generate JSON", schema=schema)
            
            assert '{"name": "test"}' in result
            
            # Verify schema was included in request
            call_args = mock_request.call_args
            body = call_args[1]["json"]
            assert body["generationConfig"]["responseMimeType"] == "application/json"


class TestChatMethod:
    """Tests for chat method"""
    
    @pytest.mark.asyncio
    async def test_chat_basic(self):
        """Test basic chat"""
        llm = GeminiLLM(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello! I'm here to help."}]
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(llm.client, 'request_async', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            messages = [Message(role="user", content="Hi")]
            result = await llm.chat(messages)
            
            assert isinstance(result, Message)
            assert result.role == "assistant"
            assert result.content == "Hello! I'm here to help."
            
            # Verify URL
            call_args = mock_request.call_args
            assert "/models/gemini-3-flash-preview:generateContent" in call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_chat_with_tools(self):
        """Test chat with tools"""
        llm = GeminiLLM(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "NYC"}
                        }
                    }]
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        
        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters=[
                ToolParam(name="city", type="string", description="City", required=True)
            ],
            function=lambda city: "Sunny"
        )
        
        with patch.object(llm.client, 'request_async', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            messages = [Message(role="user", content="What's the weather in NYC?")]
            result = await llm.chat(messages, tools=[tool])
            
            assert isinstance(result, Message)
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].name == "get_weather"
            assert result.tool_calls[0].arguments == {"city": "NYC"}


class TestEmbedMethod:
    """Tests for embed method"""
    
    @pytest.mark.asyncio
    async def test_embed_not_implemented(self):
        """Test that embed raises NotImplementedError"""
        llm = GeminiLLM(api_key="test-key")
        
        with pytest.raises(NotImplementedError) as exc_info:
            await llm.embed("test text")
        
        assert "GeminiLLM does not support embedding" in str(exc_info.value)


class TestStopMethod:
    """Tests for stop method"""
    
    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stop method calls aclose"""
        llm = GeminiLLM(api_key="test-key")
        
        with patch.object(llm.client, 'aclose', new_callable=AsyncMock) as mock_aclose:
            await llm.stop()
            mock_aclose.assert_called_once()


class TestLLMCapability:
    """Tests for LLMCapability enum"""
    
    def test_capability_values(self):
        """Test that all capabilities are defined"""
        assert LLMCapability.COMPLETION
        assert LLMCapability.CHAT
        assert LLMCapability.TOOLS
        assert LLMCapability.VISION
        assert LLMCapability.EMBEDDING
        assert LLMCapability.STREAMING

import RAW.models
import RAW.llms


def test_models_exports():
    expected_exports = {
        "Image",
        "Message",
        "ToolCall",
        "Usage",
        "Tool",
        "ToolParam",
        "LLMCapability",
        "LLMInfo",
        "jsonschema"
    }
    for item in expected_exports:
        assert hasattr(RAW.models, item), f"RAW.models is missing export: {item}"
        assert getattr(RAW.models, item) is not None


def test_llms_exports():
    expected_exports = {
        "GeminiLLM", "GeminiOptions",
        "GroqLLM", "GroqOptions",
        "VLLM", "VLLMOptions",
        "OpenAILLM", "OpenAIOptions",
        "AnthropicLLM", "AnthropicOptions",
        "BaseLLM",
    }
    for item in expected_exports:
        assert item in RAW.llms.__all__, f"RAW.llms.__all__ is missing: {item}"
        assert getattr(RAW.llms, item) is not None

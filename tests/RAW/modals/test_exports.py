import RAW.models


def test_models_exports():
    expected_exports = {
        "Image",
        "Message",
        "ToolCall",
        "Tool",
        "ToolParam",
        "LLMCapability",
        "LLMInfo",
        "jsonschema"
    }
    for item in expected_exports:
        assert hasattr(RAW.models, item), f"RAW.models is missing export: {item}"
        assert getattr(RAW.models, item) is not None

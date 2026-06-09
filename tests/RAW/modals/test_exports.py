import RAW.modals


def test_modals_exports():
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
        assert hasattr(RAW.modals, item), f"RAW.modals is missing export: {item}"
        assert getattr(RAW.modals, item) is not None

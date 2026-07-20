import pytest
from RAW.models.tools import Tool, ToolParam


def test_tool_param_to_property():
    # Basic parameter
    param1 = ToolParam(name="query", type="string", description="search query", required=True)
    prop1 = param1.to_property()
    assert prop1 == {"type": "string", "description": "search query"}

    # Parameter with enum and items
    param2 = ToolParam(
        name="tags",
        type="array",
        description="list of tags",
        enums=["news", "sports"],
        items={"type": "string"},
        required=False
    )
    prop2 = param2.to_property()
    assert prop2 == {
        "type": "array",
        "description": "list of tags",
        "enum": ["news", "sports"],
        "items": {"type": "string"}
    }


def test_tool_to_dict():
    def dummy_func(query: str) -> str:
        return f"result: {query}"

    param = ToolParam(name="query", type="string", description="search query", required=True)
    tool = Tool(
        name="web_search",
        description="searches the web",
        parameters=[param],
        function=dummy_func
    )

    assert tool.name == "web_search"
    assert tool.description == "searches the web"
    assert tool.function == dummy_func

    tool_dict = tool.to_dict()
    assert tool_dict["type"] == "function"
    assert tool_dict["function"]["name"] == "web_search"
    assert tool_dict["function"]["description"] == "searches the web"
    assert tool_dict["function"]["parameters"]["type"] == "object"
    assert "query" in tool_dict["function"]["parameters"]["properties"]
    assert tool_dict["function"]["parameters"]["properties"]["query"]["type"] == "string"
    assert tool_dict["function"]["parameters"]["required"] == ["query"]

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


def test_tool_param_accepts_full_json_schema():
    # P0-5: a parameter can describe nested objects, not only flat strings.
    lines = ToolParam(
        name="lines",
        type="array",
        description="PO lines",
        required=True,
        items={
            "type": "object",
            "properties": {"item": {"type": "string"}, "qty": {"type": "number"}},
            "required": ["item", "qty"],
        },
        json_schema={"minItems": 1},
    )
    tool = Tool(name="create_po_draft", description="d", parameters=[lines], function=lambda lines: "ok")

    schema = tool.parameters_schema()
    assert schema["properties"]["lines"]["items"]["properties"]["qty"] == {"type": "number"}
    assert schema["properties"]["lines"]["minItems"] == 1
    assert schema["required"] == ["lines"]
    assert tool.to_dict()["function"]["parameters"] == schema

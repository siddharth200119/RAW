import pytest
from pydantic import ValidationError
from RAW.modals.message import Message, ToolCall
from RAW.modals.image import Image
import numpy as np


def test_tool_call_instantiation():
    tool_call = ToolCall(id="call_abc123", name="calculator", arguments={"a": 1, "b": 2})
    assert tool_call.id == "call_abc123"
    assert tool_call.name == "calculator"
    assert tool_call.arguments == {"a": 1, "b": 2}


def test_message_instantiation_and_validation():
    # Valid message
    msg = Message(role="user", content="Hello world")
    assert msg.role == "user"
    assert msg.content == "Hello world"
    assert msg.images == []
    assert msg.tool_calls == []

    # Invalid role should raise ValidationError
    with pytest.raises(ValidationError):
        Message(role="invalid_role", content="Hello")


def test_message_serialization_deserialization():
    arr = np.zeros((5, 5, 3), dtype=np.uint8)
    img = Image.from_array(arr)
    tool_call = ToolCall(id="c1", name="add", arguments={"x": 5})

    msg = Message(
        role="assistant",
        content="Response",
        think_content="Thinking...",
        images=[img],
        tool_calls=[tool_call],
        tool_call_id="c0"
    )

    serialized = msg.to_dict()
    assert serialized["role"] == "assistant"
    assert serialized["content"] == "Response"
    assert serialized["think_content"] == "Thinking..."
    assert len(serialized["images"]) == 1
    assert serialized["images"][0] == img.to_base64()
    assert len(serialized["tool_calls"]) == 1
    assert serialized["tool_calls"][0]["name"] == "add"
    assert serialized["tool_call_id"] == "c0"

    # Deserialize back
    deserialized = Message.from_dict(serialized)
    assert deserialized.role == "assistant"
    assert deserialized.content == "Response"
    assert deserialized.think_content == "Thinking..."
    assert len(deserialized.images) == 1
    assert np.array_equal(deserialized.images[0].data, arr)
    assert len(deserialized.tool_calls) == 1
    assert deserialized.tool_calls[0].name == "add"
    assert deserialized.tool_call_id == "c0"

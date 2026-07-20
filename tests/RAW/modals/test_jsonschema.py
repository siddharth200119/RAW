import pytest
from RAW.models.jsonschema import jsonschema


def test_jsonschema_valid():
    valid_data = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }

    schema = jsonschema(valid_data)
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    assert schema["required"] == ["name"]


def test_jsonschema_invalid():
    invalid_data = {
        "type": "invalid_type_name_123"
    }

    with pytest.raises(ValueError, match="Invalid JSON Schema"):
        jsonschema(invalid_data)


def test_jsonschema_from_dict():
    valid_data = {
        "type": "string"
    }
    schema = jsonschema.from_dict(valid_data)
    assert isinstance(schema, jsonschema)
    assert schema["type"] == "string"

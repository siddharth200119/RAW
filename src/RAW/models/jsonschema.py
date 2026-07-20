import jsonschema as js_lib
from typing import Any, Dict


class jsonschema(dict):
    """A dictionary subclass that validates itself as a valid JSON Schema."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            # Validate that this dictionary is a valid JSON schema
            js_lib.Draft7Validator.check_schema(self)
        except Exception as e:
            raise ValueError(f"Invalid JSON Schema: {str(e)}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "jsonschema":
        return cls(data)

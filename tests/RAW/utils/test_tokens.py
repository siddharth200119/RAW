import builtins
import sys
import types
import pytest
from RAW.models import Message
from RAW.utils.tokens import count_tokens


@pytest.fixture
def fake_tiktoken(monkeypatch):
    """Offline stand-in for tiktoken: one token per whitespace-separated word."""
    calls = {}

    class Encoding:
        def __init__(self, name):
            self.name = name

        def encode(self, text):
            return text.split()

    def encoding_for_model(model):
        calls["model"] = model
        if model != "gpt-4o":
            raise KeyError(model)
        return Encoding("o200k_base")

    def get_encoding(name):
        calls["fallback"] = name
        return Encoding(name)

    module = types.SimpleNamespace(encoding_for_model=encoding_for_model, get_encoding=get_encoding)
    monkeypatch.setitem(sys.modules, "tiktoken", module)
    return calls


@pytest.fixture
def no_tiktoken(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_known_model_uses_model_encoding(fake_tiktoken):
    assert count_tokens("one two three", model="gpt-4o") == 3
    assert fake_tiktoken == {"model": "gpt-4o"}


def test_unknown_model_falls_back_to_cl100k(fake_tiktoken):
    assert count_tokens("one two", model="some-local-model") == 2
    assert fake_tiktoken["fallback"] == "cl100k_base"


def test_messages_are_concatenated(fake_tiktoken):
    messages = [
        Message(role="user", content="hello there "),
        Message(role="assistant", content=None),
        Message(role="user", content="general kenobi"),
    ]
    assert count_tokens(messages, model="gpt-4o") == 4


def test_without_tiktoken_estimates_four_chars_per_token(no_tiktoken):
    assert count_tokens("a" * 17) == 4
    assert count_tokens([Message(role="user", content="abcdefgh")]) == 2


def test_empty_input(no_tiktoken):
    assert count_tokens("") == 0
    assert count_tokens([]) == 0

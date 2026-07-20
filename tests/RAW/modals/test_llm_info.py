from RAW.models.llm_info import LLMInfo
from RAW.models.llm_capability import LLMCapability


def test_llm_info_instantiation():
    info = LLMInfo(
        model_name="gpt-4o",
        provider="openai",
        max_tokens=4096,
        context_window=128000,
        capabilities=[LLMCapability.CHAT, LLMCapability.TOOLS],
        metadata={"release_year": 2024}
    )

    assert info.model_name == "gpt-4o"
    assert info.provider == "openai"
    assert info.max_tokens == 4096
    assert info.context_window == 128000
    assert LLMCapability.CHAT in info.capabilities
    assert info.metadata == {"release_year": 2024}

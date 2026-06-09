from RAW.modals.llm_capability import LLMCapability


def test_llm_capability_enum():
    # Verify expected members exist
    expected_members = {
        "COMPLETION",
        "CHAT",
        "TOOLS",
        "VISION",
        "EMBEDDING",
        "STREAMING",
    }
    actual_members = {member.name for member in LLMCapability}
    assert expected_members.issubset(actual_members)

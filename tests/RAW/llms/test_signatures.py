import inspect
from RAW.llms.base import BaseLLM
from RAW.llms.vllm import VLLM


def verify_signature(target_func, expected_params):
    """Helper to verify parameter names, orders, and defaults of a function."""
    sig = inspect.signature(target_func)
    params = list(sig.parameters.values())
    
    assert len(params) == len(expected_params), f"Parameter count mismatch. Expected: {expected_params}, Got: {list(sig.parameters.keys())}"
    
    for param, expected in zip(params, expected_params):
        name, default = expected
        assert param.name == name, f"Parameter name mismatch. Expected: {name}, Got: {param.name}"
        if default is not inspect.Parameter.empty:
            assert param.default == default, f"Parameter default value mismatch for {name}. Expected: {default}, Got: {param.default}"


def test_base_llm_method_signatures():
    # 1. generate
    verify_signature(
        BaseLLM.generate,
        [
            ("self", inspect.Parameter.empty),
            ("prompt", inspect.Parameter.empty),
            ("images", None),
            ("schema", None),
            ("stream", False)
        ]
    )
    
    # 2. chat
    verify_signature(
        BaseLLM.chat,
        [
            ("self", inspect.Parameter.empty),
            ("messages", inspect.Parameter.empty),
            ("schema", None),
            ("stream", False),
            ("tools", None)
        ]
    )

    # 3. stop
    verify_signature(
        BaseLLM.stop,
        [
            ("self", inspect.Parameter.empty)
        ]
    )

    # 4. embed
    verify_signature(
        BaseLLM.embed,
        [
            ("self", inspect.Parameter.empty),
            ("text", inspect.Parameter.empty)
        ]
    )

    # 5. info
    verify_signature(
        BaseLLM.info,
        [
            ("self", inspect.Parameter.empty)
        ]
    )

    # 6. count_tokens
    verify_signature(
        BaseLLM.count_tokens,
        [
            ("self", inspect.Parameter.empty),
            ("text_or_messages", inspect.Parameter.empty)
        ]
    )

    # 7. transcribe
    verify_signature(
        BaseLLM.transcribe,
        [
            ("self", inspect.Parameter.empty),
            ("audio_bytes", inspect.Parameter.empty),
            ("mime_type", inspect.Parameter.empty)
        ]
    )

    # 8. speak
    verify_signature(
        BaseLLM.speak,
        [
            ("self", inspect.Parameter.empty),
            ("text", inspect.Parameter.empty),
            ("voice", None)
        ]
    )

    # 9. batch_generate
    verify_signature(
        BaseLLM.batch_generate,
        [
            ("self", inspect.Parameter.empty),
            ("prompts", inspect.Parameter.empty),
            ("images_list", None),
            ("schema", None)
        ]
    )


def test_vllm_signature_conformance():
    """Verify that VLLM subclass implements the same public methods with matching signatures."""
    base_methods = [m for m in dir(BaseLLM) if not m.startswith("_")]
    
    for method_name in base_methods:
        base_func = getattr(BaseLLM, method_name)
        if not inspect.iscoroutinefunction(base_func) and not inspect.isfunction(base_func):
            continue
            
        assert hasattr(VLLM, method_name), f"VLLM is missing method: {method_name}"
        
        # Check signature matches
        base_sig = inspect.signature(base_func)
        sub_sig = inspect.signature(getattr(VLLM, method_name))
        
        base_params = list(base_sig.parameters.values())
        sub_params = list(sub_sig.parameters.values())
        
        # Subclasses can support extra optional parameters, but all base parameters must match
        assert len(sub_params) >= len(base_params), f"VLLM.{method_name} has fewer parameters than base method."
        
        for bp, sp in zip(base_params, sub_params):
            assert bp.name == sp.name, f"Parameter mismatch in VLLM.{method_name}. Expected {bp.name}, got {sp.name}"
            # Verify default values match if specified in base
            if bp.default is not inspect.Parameter.empty:
                assert bp.default == sp.default, f"Parameter default value mismatch in VLLM.{method_name} for '{bp.name}'"

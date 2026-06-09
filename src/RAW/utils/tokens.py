from typing import List, Union
from RAW.modals.message import Message


def count_tokens(text_or_messages: Union[str, List[Message]], model: str = "") -> int:
    """
    Count the number of tokens in a string or list of Messages.
    Uses tiktoken if installed, otherwise falls back to estimating 4 characters per token.
    """
    if isinstance(text_or_messages, str):
        prompt = text_or_messages
    else:
        prompt = "".join([m.content or "" for m in text_or_messages])
        
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(prompt))
    except ImportError:
        # Fallback estimation (approx 4 chars per token)
        return len(prompt) // 4

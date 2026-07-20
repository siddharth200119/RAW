from abc import ABC
from RAW.models import Message, Image, LLMInfo, jsonschema, Tool
from typing import List, Union, AsyncGenerator, Optional, Dict
import numpy as np


class BaseLLM(ABC):

    async def generate(
        self,
        prompt: str,
        images: Optional[List[Image]] = None,
        schema: Optional[jsonschema] = None,
        stream: bool = False
    ) -> Union[str, Dict, AsyncGenerator[Union[str, Dict], None]]:
        raise NotImplementedError("This LLM provider does not implement the generate method.")

    async def chat(
        self, 
        messages: List[Message],
        schema: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Tool]] = None
    ) -> Union[Message, AsyncGenerator[Message, None]]:
        raise NotImplementedError("This LLM provider does not implement the chat method.")

    async def stop(self) -> bool:
        raise NotImplementedError("This LLM provider does not implement the stop method.")

    async def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError("This LLM provider does not implement the embed method.")

    def info(self) -> LLMInfo:
        raise NotImplementedError("This LLM provider does not implement the info method.")

    async def count_tokens(self, text_or_messages: Union[str, List[Message]]) -> int:
        raise NotImplementedError("This LLM provider does not implement token counting.")

    async def transcribe(self, audio_bytes: bytes, mime_type: str) -> str:
        raise NotImplementedError("This LLM provider does not implement audio transcription.")

    async def speak(self, text: str, voice: Optional[str] = None) -> bytes:
        raise NotImplementedError("This LLM provider does not implement text-to-speech.")

    async def batch_generate(
        self,
        prompts: List[str],
        images_list: Optional[List[Optional[List[Image]]]] = None,
        schema: Optional[jsonschema] = None
    ) -> List[Union[str, Dict]]:
        """Default batch generation using parallel async generate calls."""
        import asyncio
        if images_list is None:
            images_list = [None] * len(prompts)
        elif len(images_list) != len(prompts):
            raise ValueError("images_list must have the same length as prompts.")

        tasks = [
            self.generate(prompt=prompt, images=images, schema=schema, stream=False)
            for prompt, images in zip(prompts, images_list)
        ]
        return list(await asyncio.gather(*tasks))
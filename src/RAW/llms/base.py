from abc import ABC, abstractmethod
from RAW.modals import Message
from typing import List, Union, AsyncGenerator
import jsonschema
import numpy as np

class BaseLLM(ABC):

    @abstractmethod
    async def chat(self, messages: List[Message], schema: jsonschema = None, stream: bool = False) -> Union[Message, AsyncGenerator[Message, None]]:
        pass

    @abstractmethod
    async def stop(self) -> bool:
        pass

    @abstractmethod
    async def embed(self,text:str) -> np.ndarray:
        pass
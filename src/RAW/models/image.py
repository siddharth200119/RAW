import cv2
import numpy as np
import base64
from typing import Union
from pydantic import BaseModel
from pathlib import Path


class Image(BaseModel):
    data: np.ndarray
    mime_type: str = "image/jpeg"

    model_config = {
        "arbitrary_types_allowed": True  # This allows numpy arrays
    }

    @classmethod
    def from_file(cls, file_path: Union[str, Path]):
        img = cv2.imread(str(file_path))
        if img is None:
            raise ValueError("Image could not be loaded from the file.")
        return cls(data=img)

    @classmethod
    def from_bytes(cls, image_bytes: bytes):
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image could not be decoded from bytes.")
        return cls(data=img)

    @classmethod
    def from_base64(cls, base64_string: str):
        header_removed = base64_string.split(',')[-1]
        image_bytes = base64.b64decode(header_removed)
        return cls.from_bytes(image_bytes)

    @classmethod
    def from_array(cls, array: np.ndarray):
        return cls(data=array)

    def to_bytes(self, ext: str = ".jpg") -> bytes:
        success, buffer = cv2.imencode(ext, self.data)
        if not success:
            raise ValueError("Image could not be encoded to bytes.")
        return buffer.tobytes()

    def to_base64(self, ext: str = ".jpg") -> str:
        image_bytes = self.to_bytes(ext)
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        # return f"data:image/{ext.strip('.')};base64,{base64_str}"
        return base64_str

    def to_array(self) -> np.ndarray:
        return self.data

    def save(self, file_path: Union[str, Path]):
        cv2.imwrite(str(file_path), self.data)

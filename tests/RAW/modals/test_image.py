import numpy as np
import pytest
import cv2
import base64
from pathlib import Path
from RAW.modals.image import Image


def test_image_from_array():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    # Set a custom pixel value to check integrity
    arr[0, 0] = [255, 0, 0]

    img = Image.from_array(arr)
    assert isinstance(img, Image)
    assert np.array_equal(img.data, arr)
    assert np.array_equal(img.to_array(), arr)


def test_image_to_bytes_and_base64():
    arr = np.zeros((5, 5, 3), dtype=np.uint8)
    img = Image.from_array(arr)

    # Convert to bytes
    img_bytes = img.to_bytes()
    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 0

    # Convert to base64
    img_b64 = img.to_base64()
    assert isinstance(img_b64, str)
    assert len(img_b64) > 0

    # Decode bytes back and verify
    decoded_bytes = base64.b64decode(img_b64)
    assert decoded_bytes == img_bytes


def test_image_from_bytes_and_base64():
    # Create valid JPEG bytes
    arr = np.ones((8, 8, 3), dtype=np.uint8) * 128
    _, jpeg_bytes = cv2.imencode(".jpg", arr)
    jpeg_bytes = jpeg_bytes.tobytes()

    # Load from bytes
    img_from_bytes = Image.from_bytes(jpeg_bytes)
    assert img_from_bytes.data.shape == (8, 8, 3)

    # Load from base64
    b64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
    img_from_b64 = Image.from_base64(b64_str)
    assert img_from_b64.data.shape == (8, 8, 3)

    # Load from base64 with data URL header (from_base64 uses split(','))
    b64_url = f"data:image/jpeg;base64,{b64_str}"
    img_from_b64_url = Image.from_base64(b64_url)
    assert img_from_b64_url.data.shape == (8, 8, 3)


def test_image_save_and_from_file(tmp_path):
    arr = np.zeros((6, 6, 3), dtype=np.uint8)
    img = Image.from_array(arr)

    file_path = tmp_path / "test_img.jpg"
    img.save(file_path)

    assert file_path.exists()

    # Load it back from file
    loaded_img = Image.from_file(file_path)
    assert loaded_img.data.shape == (6, 6, 3)


def test_image_error_handling():
    # Test loading from non-existent file
    with pytest.raises(ValueError, match="Image could not be loaded"):
        Image.from_file("non_existent_file_path_12345.jpg")

    # Test decoding invalid bytes
    with pytest.raises(ValueError, match="Image could not be decoded"):
        Image.from_bytes(b"invalid jpeg bytes")

import io

import numpy as np
import pytest
from PIL import Image

from server.model_service import InvalidImageError, prepare


def png_bytes(size=(64, 64), color=(120, 120, 120)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_prepare_returns_batched_float_tensor_in_unit_range():
    prepared = prepare(png_bytes())
    assert prepared.tensor.shape == (1, 224, 224, 3)
    assert prepared.tensor.dtype == np.float32
    assert float(prepared.tensor.min()) >= 0.0
    assert float(prepared.tensor.max()) <= 1.0


def test_prepare_keeps_original_size_and_resizes_to_224():
    prepared = prepare(png_bytes(size=(300, 200)))
    assert prepared.original.size == (300, 200)
    assert prepared.resized.size == (224, 224)


def test_prepare_scales_pixels_to_zero_one():
    prepared = prepare(png_bytes(size=(224, 224), color=(255, 255, 255)))
    assert float(prepared.tensor.max()) == pytest.approx(1.0, abs=1e-6)
    assert float(prepared.tensor.min()) == pytest.approx(1.0, abs=1e-6)


def test_prepare_rejects_non_image_bytes():
    with pytest.raises(InvalidImageError):
        prepare(b"not-an-image-at-all")


def test_prepare_rejects_empty_payload():
    with pytest.raises(InvalidImageError):
        prepare(b"")

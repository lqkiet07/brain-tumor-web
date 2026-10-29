import io

import numpy as np
from PIL import Image

from server.model_service import heat_overlay, prepare


def png_bytes(size=(64, 64), color=(120, 120, 120)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_overlay_keeps_original_resolution():
    prepared = prepare(png_bytes(size=(300, 200)))
    overlay = heat_overlay(prepared, np.zeros((14, 14), dtype=np.float32))
    assert overlay.size == (300, 200)
    assert overlay.mode == "RGB"


def test_overlay_applies_documented_formula():
    """Overlay = 0.6 * anh_goc + 0.4 * jet(cam), dung cong thuc trong bao cao."""
    prepared = prepare(png_bytes(size=(224, 224), color=(10, 20, 30)))
    cam = np.zeros((14, 14), dtype=np.float32)
    overlay = np.asarray(heat_overlay(prepared, cam)).astype(float)
    base = np.asarray(prepared.resized).astype(float)
    jet_zero = np.array([0.0, 0.0, 127.5])  # matplotlib jet(0) = xanh dam
    expected = np.clip(base * 0.6 + jet_zero * 0.4, 0, 255)
    assert np.abs(overlay - expected).max() <= 2


def test_overlay_with_hotspot_brightens_pixels():
    prepared = prepare(png_bytes(size=(224, 224), color=(0, 0, 0)))
    cam = np.zeros((14, 14), dtype=np.float32)
    cam[7, 7] = 1.0
    overlay = np.asarray(heat_overlay(prepared, cam))
    assert int(overlay.max()) > 0


def test_overlay_never_exceeds_uint8_range():
    prepared = prepare(png_bytes(size=(224, 224), color=(255, 255, 255)))
    cam = np.ones((14, 14), dtype=np.float32)
    overlay = np.asarray(heat_overlay(prepared, cam))
    assert overlay.dtype == np.uint8
    assert int(overlay.max()) <= 255

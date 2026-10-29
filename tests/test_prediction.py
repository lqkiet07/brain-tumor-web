import base64
import io

import pytest
from PIL import Image

pytest.importorskip("tensorflow")

from server.config import FIXTURES_DIR  # noqa: E402
from server.model_service import ModelService  # noqa: E402


@pytest.fixture(scope="module")
def service():
    return ModelService()


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _fixtures():
    return sorted(FIXTURES_DIR.glob("tile_*.png"))


def test_nine_fixtures_exist():
    assert len(_fixtures()) == 9


def test_predict_returns_probabilities_that_sum_to_one(service):
    with Image.open(_fixtures()[0]) as image:
        result = service.predict(_png_bytes(image))

    assert len(result["probabilities"]) == 4
    total = sum(item["probability"] for item in result["probabilities"])
    assert total == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["predicted_class"] in {"glioma", "meningioma", "notumor", "pituitary"}


def test_all_nine_augmentations_predict_the_same_class(service):
    predictions = []
    for path in _fixtures():
        with Image.open(path) as image:
            predictions.append(service.predict(_png_bytes(image))["predicted_class"])
    assert len(set(predictions)) == 1, f"Các biến thể augmentation cho khác kết quả: {predictions}"


def test_heatmap_overlay_matches_uploaded_image_size(service):
    with Image.open(_fixtures()[0]) as image:
        expected_size = image.size
        result = service.predict(_png_bytes(image))

    assert result["heatmap_layer"] == "block5_conv3"
    overlay = Image.open(io.BytesIO(base64.b64decode(result["heatmap_png_base64"])))
    assert overlay.size == expected_size


def test_predict_reports_timing_and_model_metadata(service):
    with Image.open(_fixtures()[0]) as image:
        result = service.predict(_png_bytes(image))

    assert result["predict_ms"] > 0
    assert result["total_ms"] >= result["predict_ms"]
    assert result["model"]["total_params"] == 14_716_740
    assert result["model"]["trainable_params"] == 12_981_252

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server.app import app

client = TestClient(app)


def _png_bytes(size=(64, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (90, 110, 130)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_model_metadata():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["default_model"] == "vgg16_b4b5"
    assert body["models_count"] >= 3


def test_get_models_list():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    model_ids = [m["id"] for m in models]
    assert "vgg16_b4b5" in model_ids
    assert "resnet50" in model_ids
    assert "vgg16_block5" in model_ids
    resnet_info = next(m for m in models if m["id"] == "resnet50")
    assert resnet_info["accuracy"] == 93.00
    assert resnet_info["track"] == "resnet"


def test_predict_rejects_non_image_upload():
    response = client.post("/api/predict", files={"file": ("note.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "ảnh" in response.json()["detail"].lower()


def test_predict_rejects_missing_file():
    response = client.post("/api/predict")
    assert response.status_code == 422


def test_predict_rejects_oversized_upload():
    payload = b"\x00" * (11 * 1024 * 1024)
    response = client.post("/api/predict", files={"file": ("big.png", payload, "image/png")})
    assert response.status_code == 413


def test_predict_returns_full_result_for_valid_image():
    from server.config import FIXTURES_DIR

    fixture = sorted(FIXTURES_DIR.glob("tile_*.png"))[0]
    with Image.open(fixture) as image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

    response = client.post("/api/predict", files={"file": ("mri.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in {"glioma", "meningioma", "notumor", "pituitary"}
    assert len(body["probabilities"]) == 4
    assert sum(item["probability"] for item in body["probabilities"]) == pytest.approx(1.0, abs=1e-3)
    assert body["heatmap_png_base64"]
    assert body["model"]["total_params"] == 14_716_740


def test_predict_with_resnet50_model():
    from server.config import FIXTURES_DIR

    fixture = sorted(FIXTURES_DIR.glob("tile_*.png"))[0]
    with Image.open(fixture) as image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

    response = client.post("/api/predict?model=resnet50", files={"file": ("mri.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "notumor"
    assert body["model"]["id"] == "resnet50"
    assert body["model"]["total_params"] == 23_595_908
    assert body["heatmap_layer"] == "conv5_block3_out"


def test_predict_rejects_unknown_model_id():
    from server.config import FIXTURES_DIR

    fixture = sorted(FIXTURES_DIR.glob("tile_*.png"))[0]
    with Image.open(fixture) as image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

    response = client.post("/api/predict?model=unknown_net", files={"file": ("mri.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 404
    assert "không tồn tại" in response.json()["detail"].lower()


def test_static_site_is_served_at_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

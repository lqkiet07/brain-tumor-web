from pathlib import Path
from server.config import BASE_DIR, MODEL_REGISTRY


def test_every_registry_model_lives_inside_models_dir():
    models_dir = (BASE_DIR / "models").resolve()
    assert models_dir.exists(), f"Thu muc models khong ton tai: {models_dir}"

    for model_id, info in MODEL_REGISTRY.items():
        model_path = Path(info["path"]).resolve()
        assert model_path.parent == models_dir, (
            f"Model '{model_id}' phai nam trong thu muc models/, duong dan hien tai: {model_path}"
        )
        assert model_path.exists(), f"File model '{model_id}' khong ton tai tai: {model_path}"
        assert model_path.stat().st_size > 0, f"File model '{model_id}' bi rong: {model_path}"


def test_registry_contains_three_live_models():
    assert set(MODEL_REGISTRY.keys()) == {"vgg16_b4b5", "resnet50", "vgg16_block5"}

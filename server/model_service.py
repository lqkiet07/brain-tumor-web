"""Nạp model động theo registry, tiền xử lý đa kiến trúc và trích xuất activation map.

Lưu ý:
- VGG: RGB 224x224 chia 255.
- ResNet50: tf.keras.applications.resnet50.preprocess_input (BGR, trừ mean ImageNet).
- Bản đồ nhiệt là activation map (mean activation của layer conv đích), không phải Grad-CAM.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import (
    CLASS_LABELS_VI,
    CLASS_NAMES,
    DEFAULT_MODEL_ID,
    IMG_SIZE,
    MODEL_REGISTRY,
    OVERLAY_ALPHA,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class InvalidImageError(Exception):
    """Ảnh upload không đọc được (sai định dạng, file rỗng...)."""


class InvalidModelError(Exception):
    """Mô hình được yêu cầu không có trong registry."""


class ModelNotReadyError(Exception):
    """Mô hình chưa sẵn sàng để chạy."""


@dataclass
class Prepared:
    tensor: np.ndarray
    original: Image.Image
    resized: Image.Image


def prepare(raw: bytes, preprocess: str = "scale_255") -> Prepared:
    """Chuẩn hóa ảnh upload theo đúng pipeline lúc huấn luyện của từng mô hình."""
    if not raw:
        raise InvalidImageError("Tệp rỗng, không có dữ liệu ảnh.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise InvalidImageError("Không đọc được ảnh. Hãy chọn file JPEG hoặc PNG hợp lệ.") from exc

    original = image.convert("RGB")
    resized = original.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
    arr = np.asarray(resized, dtype=np.float32)

    if preprocess == "resnet_caffe":
        import tensorflow as tf

        # preprocess_input của ResNet chuyển RGB -> BGR và trừ vector ImageNet mean
        tensor = tf.keras.applications.resnet50.preprocess_input(arr.copy())[None, ...]
    else:
        # Mặc định VGG/MobileNet: chia 255 về [0, 1]
        tensor = (arr / 255.0)[None, ...]

    return Prepared(tensor=tensor, original=original, resized=resized)


def heat_overlay(prepared: Prepared, cam: np.ndarray, alpha: float = OVERLAY_ALPHA) -> Image.Image:
    """Chồng activation map lên ảnh gốc theo công thức 0.6·ảnh + 0.4·nhiệt."""
    import matplotlib

    cam_image = Image.fromarray((np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
    )
    heat = matplotlib.colormaps["jet"](np.asarray(cam_image, dtype=np.float32) / 255.0)[:, :, :3] * 255.0
    base = np.asarray(prepared.resized, dtype=np.float32)
    merged = np.clip(base * (1.0 - alpha) + heat * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(merged).resize(prepared.original.size, Image.Resampling.BILINEAR)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def load_model_from_path(model_path: Path):
    """Nạp file model .h5 (Keras 3)."""
    import tensorflow as tf

    path = Path(model_path)
    if not path.exists():
        raise ModelNotReadyError(f"Không tìm thấy file model tại {path}.")
    return tf.keras.models.load_model(str(path), compile=False)


def load_model(model_path: Path | None = None):
    """Alias tương thích ngược nạp model mặc định (VGG16 B4B5)."""
    target = model_path or MODEL_REGISTRY[DEFAULT_MODEL_ID]["path"]
    return load_model_from_path(target)


def find_layer(model, name: str):
    """Tìm layer theo tên, hỗ trợ cả cấu trúc layer lồng nhau."""
    for layer in model.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, "layers"):
            try:
                return layer.get_layer(name)
            except (ValueError, KeyError):
                continue
    raise KeyError(f"Không tìm thấy layer '{name}' trong model.")


def count_params(model, trainable_only: bool = True) -> int:
    weights = model.trainable_weights if trainable_only else model.weights
    return int(sum(np.prod(w.shape) for w in weights))


def activation_map(model, tensor: np.ndarray, layer_name: str) -> np.ndarray:
    """Trung bình activation của một conv layer, chuẩn hóa về [0, 1]."""
    import tensorflow as tf

    layer = find_layer(model, layer_name)
    extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
    activations = extractor.predict(tensor, verbose=0)[0]
    cam = np.maximum(activations.mean(axis=-1), 0.0)
    peak = float(cam.max())
    if peak > 0:
        cam = cam / peak
    return cam


class ModelService:
    """Bộ điều phối nạp động đa mô hình (Multi-Model Registry Dispatcher)."""

    def __init__(self, registry: dict = MODEL_REGISTRY) -> None:
        self.registry = registry
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()

    def is_loaded(self, model_id: str = DEFAULT_MODEL_ID) -> bool:
        return model_id in self._models

    def ensure_model(self, model_id: str = DEFAULT_MODEL_ID):
        if model_id not in self.registry:
            raise InvalidModelError(
                f"Mô hình '{model_id}' không hợp lệ. Khả dụng: {list(self.registry.keys())}"
            )
        if model_id not in self._models:
            with self._lock:
                if model_id not in self._models:
                    info = self.registry[model_id]
                    self._models[model_id] = load_model_from_path(info["path"])
        return self._models[model_id]

    def available_models(self) -> list[dict]:
        return [
            {
                "id": m_id,
                "name": info["name"],
                "track": info["track"],
                "researcher": info["researcher"],
                "accuracy": info["accuracy"],
                "f1": info["f1"],
                "reported_latency_ms": info["reported_latency_ms"],
                "total_params": info["total_params"],
                "badge": info["badge"],
                "loaded": m_id in self._models,
            }
            for m_id, info in self.registry.items()
        ]

    def predict(self, raw: bytes, model_id: str = DEFAULT_MODEL_ID) -> dict:
        if model_id not in self.registry:
            raise InvalidModelError(
                f"Mô hình '{model_id}' không tồn tại. Lựa chọn khả dụng: {list(self.registry.keys())}"
            )

        info = self.registry[model_id]
        model = self.ensure_model(model_id)
        prepared = prepare(raw, preprocess=info["preprocess"])

        start = time.perf_counter()
        probabilities = model.predict(prepared.tensor, verbose=0)[0]
        predict_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        if info["heatmap_layer"]:
            cam = activation_map(model, prepared.tensor, info["heatmap_layer"])
            heatmap_layer_name = info["heatmap_layer"]
        else:
            cam = np.zeros((14, 14), dtype=np.float32)
            heatmap_layer_name = "none"
        heatmap_ms = (time.perf_counter() - start) * 1000.0

        overlay = heat_overlay(prepared, cam)
        best = int(np.argmax(probabilities))

        return {
            "predicted_class": CLASS_NAMES[best],
            "predicted_label": CLASS_LABELS_VI[CLASS_NAMES[best]],
            "confidence": round(float(probabilities[best]), 4),
            "probabilities": [
                {
                    "class": name,
                    "label": CLASS_LABELS_VI[name],
                    "probability": round(float(prob), 4),
                }
                for name, prob in zip(CLASS_NAMES, probabilities)
            ],
            "predict_ms": round(predict_ms, 1),
            "heatmap_ms": round(heatmap_ms, 1),
            "total_ms": round(predict_ms + heatmap_ms, 1),
            "heatmap_layer": heatmap_layer_name,
            "heatmap_png_base64": base64.b64encode(_png_bytes(overlay)).decode("ascii"),
            "model": {
                "id": model_id,
                "name": info["name"],
                "track": info["track"],
                "researcher": info["researcher"],
                "accuracy": info["accuracy"],
                "f1": info["f1"],
                "reported_latency_ms": info["reported_latency_ms"],
                "total_params": info["total_params"],
                "trainable_params": info.get("trainable_params", info["total_params"]),
                "badge": info["badge"],
                "preprocess": info["preprocess"],
            },
        }

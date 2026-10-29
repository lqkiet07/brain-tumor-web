"""FastAPI: phục vụ trang web tĩnh + API chẩn đoán đa mô hình.

Chạy local: .venv\\Scripts\\python.exe run_demo.py -> http://127.0.0.1:8010
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_MODEL_ID, MAX_UPLOAD_BYTES, WEB_DIR
from .model_service import (
    InvalidImageError,
    InvalidModelError,
    ModelNotReadyError,
    ModelService,
)

app = FastAPI(title="Brain Tumor MRI — Multi-Model Diagnostic Lab", version="2.0.0")
service = ModelService()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL_ID,
        "models_count": len(service.registry),
        "loaded_models": [m_id for m_id in service.registry if service.is_loaded(m_id)],
    }


@app.get("/api/models")
def get_models() -> list[dict]:
    """Danh sách các mô hình khả dụng để chạy thử nghiệm trực tiếp."""
    return service.available_models()


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query(default=DEFAULT_MODEL_ID, description="ID mô hình cần chạy"),
) -> dict:
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Ảnh vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    try:
        return service.predict(raw, model_id=model)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InvalidModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# Route API khai báo trước, static mount sau để không nuốt "/api/*"
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

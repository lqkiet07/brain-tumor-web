"""Do do tre cua API du doan (co warm-up) - dung de ghi vao README."""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8010"
FIXTURE = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tile_04.png"
RUNS = 5


def post(path: str, filename: str, payload: bytes) -> dict:
    boundary = "----brain-tumor-bench"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{BASE}{path}", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> int:
    payload = FIXTURE.read_bytes()
    walls: list[float] = []
    predicts: list[float] = []

    print(f"  Warm-up + {RUNS} lần đo (máy này, CPU):")
    for index in range(RUNS + 1):
        start = time.perf_counter()
        result = post("/api/predict", "mri.png", payload)
        wall = (time.perf_counter() - start) * 1000
        tag = "warm-up" if index == 0 else f"run {index}"
        print(
            f"    {tag:8s} tổng={wall:7.1f} ms | suy luận={result['predict_ms']:6.1f} ms | "
            f"heatmap={result['heatmap_ms']:6.1f} ms | {result['predicted_class']} ({result['confidence']:.2%})"
        )
        if index:
            walls.append(wall)
            predicts.append(result["predict_ms"])

    print(
        f"\n  Trung bình sau warm-up: tổng {statistics.mean(walls):.1f} ms | "
        f"suy luận {statistics.mean(predicts):.1f} ms"
    )
    print("  (Báo cáo ghi 75,42 ms/ảnh — đo trên GPU Kaggle, không phải CPU local.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

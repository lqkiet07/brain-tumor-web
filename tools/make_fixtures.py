"""Cat 9 o tile tu anh augmentation (cung 1 anh MRI, 9 bien the augmentation).

Dung cho test pipeline: 9 tile deu phai duoc model doan cung mot nhan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE = BASE_DIR.parent / "project 1" / "report_asset" / "results_vgg16_b4b5" / "augmented_image.png"
OUT_DIR = BASE_DIR / "tests" / "fixtures"

DARK_THRESHOLD = 128
MIN_AXIS_RATIO = 0.25


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Tra ve danh sach (start, end) cua cac doan True lien tiep trong mask 1D."""
    spans: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _detect_grid(dark: np.ndarray) -> list[tuple[int, int]]:
    col_ratio = dark.mean(axis=0)
    row_ratio = dark.mean(axis=1)
    cols = _runs(col_ratio > MIN_AXIS_RATIO)
    rows = _runs(row_ratio > MIN_AXIS_RATIO)
    if len(cols) != 3 or len(rows) != 3:
        raise ValueError(f"Khong tim thay luoi 3x3 (cols={len(cols)}, rows={len(rows)})")
    return rows, cols


def _fallback_grid(size: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    width, height = size
    margin_x, margin_y = int(width * 0.02), int(height * 0.02)
    step_x = (width - 2 * margin_x) // 3
    step_y = (height - 2 * margin_y) // 3
    rows = [(margin_y + i * step_y, margin_y + (i + 1) * step_y) for i in range(3)]
    cols = [(margin_x + i * step_x, margin_x + (i + 1) * step_x) for i in range(3)]
    return rows, cols


def _trim_white(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("L"))
    mask = array < 240
    if not mask.any():
        return image
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    return image.crop((left, top, right, bottom))


def build_fixtures(source: Path = SOURCE, out_dir: Path = OUT_DIR) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Khong tim thay anh nguon: {source}")

    grid = Image.open(source).convert("RGB")
    dark = np.asarray(grid.convert("L")) < DARK_THRESHOLD
    try:
        rows, cols = _detect_grid(dark)
        strategy = "tu phat hien"
    except ValueError as exc:
        print(f"  [canh bao] {exc} -> dung chia deu 3x3")
        rows, cols = _fallback_grid(grid.size)
        strategy = "chia deu 3x3"

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index = 0
    for top, bottom in rows:
        for left, right in cols:
            tile = _trim_white(grid.crop((left, top, right, bottom)))
            path = out_dir / f"tile_{index:02d}.png"
            tile.save(path, optimize=True)
            written.append(path)
            index += 1

    print(f"  Nguon : {source.name} ({grid.size[0]}x{grid.size[1]}) - cach {strategy}")
    print(f"  Da ghi {len(written)} tile vao {out_dir}")
    return written


if __name__ == "__main__":
    paths = build_fixtures()
    if len(paths) != 9:
        print(f"Loi: mong doi 9 tile, nhan duoc {len(paths)}", file=sys.stderr)
        raise SystemExit(1)
    for path in paths:
        with Image.open(path) as image:
            print(f"    {path.name}: {image.size[0]}x{image.size[1]} {image.mode}")

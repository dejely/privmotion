from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_pgm(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.astype(np.uint8)
    if array.ndim != 2:
        raise ValueError("PGM export expects a single-channel image")
    height, width = array.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + array.tobytes())


def depth_surrogate_from_frame(image: np.ndarray, max_size: int = 64, levels: int = 16) -> np.ndarray:
    if image.ndim == 3:
        gray = (
            0.299 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]
        ).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    height, width = gray.shape
    scale = min(max_size / max(height, width), 1.0)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))

    y_idx = np.linspace(0, height - 1, target_height).round().astype(int)
    x_idx = np.linspace(0, width - 1, target_width).round().astype(int)
    resized = gray[np.ix_(y_idx, x_idx)]

    bucket = max(1, 256 // levels)
    return ((resized // bucket) * bucket).astype(np.uint8)


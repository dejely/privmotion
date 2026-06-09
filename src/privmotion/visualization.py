from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

from privmotion.io import read_portable_anymap


@dataclass(frozen=True)
class VisualizationResult:
    output_dir: Path
    visualization_path: Path
    frame_count: int
    frames_dir: Path | None
    raw_rgb_used: bool = False


SKELETON_EDGES = (
    ("head", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("neck", "hip_center"),
    ("hip_center", "left_ankle"),
    ("hip_center", "right_ankle"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def visualize_output_dir(
    output_dir: Path,
    visualization_path: Path,
    frames_dir: Path | None = None,
    fps: int = 4,
    size: tuple[int, int] = (640, 360),
) -> VisualizationResult:
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"output directory does not exist: {output_path}")
    if not output_path.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_path}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("visualization size must be positive")

    metadata_path = output_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"required metadata.json is missing from: {output_path}")

    metadata = _read_json(metadata_path)
    if _deidentification_profile(metadata) == "hipaa-expert-aggregate":
        raise ValueError("hipaa-expert-aggregate outputs do not support visualization")
    skeletons = _read_optional_json(output_path / "skeletons.json")
    retention = _read_optional_json(output_path / "retention_report.json")
    skeleton_by_frame = _skeletons_by_frame(skeletons)
    silhouette_files = _frame_files(output_path / "silhouettes")
    depth_files = _frame_files(output_path / "depth_surrogates")
    frame_indices = _frame_indices(metadata, skeleton_by_frame, silhouette_files, depth_files)

    frames = [
        _render_frame(
            frame_index=index,
            size=size,
            metadata=metadata,
            retention=retention,
            skeleton_records=skeleton_by_frame.get(index, []),
            silhouette_path=silhouette_files.get(index),
            depth_path=depth_files.get(index),
        )
        for index in frame_indices
    ]

    if not frames:
        frames = [
            _render_frame(
                frame_index=0,
                size=size,
                metadata=metadata,
                retention=retention,
                skeleton_records=[],
                silhouette_path=None,
                depth_path=None,
            )
        ]

    visualization = Path(visualization_path)
    visualization.parent.mkdir(parents=True, exist_ok=True)
    _write_visualization(visualization, frames, fps)

    png_dir = Path(frames_dir) if frames_dir is not None else None
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)
        for offset, frame in enumerate(frames):
            frame.save(png_dir / f"frame_{frame_indices[offset]:06d}.png")

    return VisualizationResult(
        output_dir=output_path,
        visualization_path=visualization,
        frame_count=len(frames),
        frames_dir=png_dir,
        raw_rgb_used=False,
    )


def _write_visualization(path: Path, frames: list[Image.Image], fps: int) -> None:
    suffix = path.suffix.lower()
    if suffix == ".gif":
        duration_ms = max(1, int(1000 / fps))
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return

    if suffix == ".mp4":
        array = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
        iio.imwrite(path, array, fps=fps, macro_block_size=1)
        return

    raise ValueError("visualization path must end with .gif or .mp4")


def _render_frame(
    frame_index: int,
    size: tuple[int, int],
    metadata: dict[str, Any],
    retention: dict[str, Any] | None,
    skeleton_records: list[dict[str, Any]],
    silhouette_path: Path | None,
    depth_path: Path | None,
) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", size, (18, 23, 31))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title_h = 46
    footer_h = 34
    margin = 12
    panel_gap = 10
    panel_h = max(1, height - title_h - footer_h - margin * 2)
    panel_w = max(1, (width - margin * 2 - panel_gap) // 2)
    left_box = (margin, title_h + margin, margin + panel_w, title_h + margin + panel_h)
    right_box = (margin + panel_w + panel_gap, title_h + margin, width - margin, title_h + margin + panel_h)

    retention_passed = _retention_passed(retention)
    retention_label = "retention: pass" if retention_passed else "retention: unavailable/fail"
    artifacts = _artifact_label(silhouette_path, depth_path, skeleton_records)
    draw.text((margin, 12), f"privmotion anonymized preview | frame {frame_index:06d}", fill=(235, 240, 245), font=font)
    draw.text((width - 190, 12), retention_label, fill=(146, 230, 178) if retention_passed else (255, 191, 115), font=font)

    _draw_panel(draw, left_box, "silhouette + skeleton")
    _draw_panel(draw, right_box, "depth surrogate")

    silhouette = _load_pgm(silhouette_path)
    depth = _load_pgm(depth_path)

    reference_shape: tuple[int, int] | None = None
    if silhouette is not None:
        reference_shape = silhouette.shape
        _paste_gray_panel(canvas, silhouette, left_box, tint=(74, 214, 160))
    else:
        _draw_center_text(draw, left_box, "no silhouette", font)

    if depth is not None:
        if reference_shape is None:
            reference_shape = depth.shape
        _paste_gray_panel(canvas, depth, right_box, tint=(140, 164, 255))
    else:
        _draw_center_text(draw, right_box, "no depth surrogate", font)

    if reference_shape is None:
        reference_shape = _shape_from_skeletons(skeleton_records) or (100, 100)
    _draw_skeleton(draw, left_box, reference_shape, skeleton_records, font)

    frame_count = int(metadata.get("input_summary", {}).get("readable_frames", 0) or 0)
    track_ids = sorted({record.get("track_id") for record in skeleton_records if record.get("track_id") is not None})
    footer = f"frames: {frame_count} | tracks: {track_ids or 'none'} | artifacts: {artifacts} | raw_rgb_used: false"
    draw.text((margin, height - footer_h + 10), footer, fill=(206, 213, 224), font=font)
    return canvas


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _deidentification_profile(metadata: dict[str, Any]) -> str:
    profile = metadata.get("deidentification_profile")
    if isinstance(profile, str):
        return profile
    config = metadata.get("config", {})
    if isinstance(config, dict) and isinstance(config.get("deidentification_profile"), str):
        return str(config["deidentification_profile"])
    return "standard"


def _skeletons_by_frame(payload: dict[str, Any] | None) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    if payload is None:
        return by_frame
    for record in payload.get("records", []):
        if not isinstance(record, dict):
            continue
        frame_index = int(record.get("frame_index", 0) or 0)
        by_frame.setdefault(frame_index, []).append(record)
    return by_frame


def _frame_files(path: Path) -> dict[int, Path]:
    if not path.exists():
        return {}
    files: dict[int, Path] = {}
    for child in sorted(path.glob("frame_*.pgm")):
        match = re.search(r"frame_(\d+)\.pgm$", child.name)
        if match:
            files[int(match.group(1))] = child
    return files


def _frame_indices(
    metadata: dict[str, Any],
    skeletons: dict[int, list[dict[str, Any]]],
    silhouettes: dict[int, Path],
    depths: dict[int, Path],
) -> list[int]:
    indices = set(skeletons) | set(silhouettes) | set(depths)
    if indices:
        return sorted(indices)
    count = int(metadata.get("input_summary", {}).get("readable_frames", 0) or 0)
    return list(range(count))


def _load_pgm(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    image = read_portable_anymap(path)
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim == 3:
        return array[:, :, 0]
    return array


def _draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str) -> None:
    draw.rounded_rectangle(box, radius=6, outline=(69, 79, 94), fill=(28, 35, 46), width=1)
    draw.text((box[0] + 8, box[1] + 8), label, fill=(215, 222, 232), font=ImageFont.load_default())


def _paste_gray_panel(
    canvas: Image.Image,
    image: np.ndarray,
    box: tuple[int, int, int, int],
    tint: tuple[int, int, int],
) -> None:
    target = _fit_box(image.shape[1], image.shape[0], box, top_padding=26)
    normalized = image.astype(np.float32) / 255.0
    rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for channel, value in enumerate(tint):
        rgb[:, :, channel] = np.clip(normalized * value, 0, 255).astype(np.uint8)
    panel = Image.fromarray(rgb).resize((target[2] - target[0], target[3] - target[1]), Image.Resampling.NEAREST)
    canvas.paste(panel, target)


def _draw_skeleton(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    reference_shape: tuple[int, int],
    records: list[dict[str, Any]],
    font: ImageFont.ImageFont,
) -> None:
    if not records:
        return
    ref_h, ref_w = reference_shape
    target = _fit_box(ref_w, ref_h, box, top_padding=26)

    for record in records:
        points = {
            point.get("name"): point
            for point in record.get("keypoints", [])
            if isinstance(point, dict) and point.get("name") is not None
        }
        mapped = {
            name: _map_point(float(point.get("x", 0.0)), float(point.get("y", 0.0)), ref_w, ref_h, target)
            for name, point in points.items()
        }
        for start, end in SKELETON_EDGES:
            if start in mapped and end in mapped:
                draw.line((mapped[start], mapped[end]), fill=(255, 210, 96), width=2)
        for name, xy in mapped.items():
            r = 3
            draw.ellipse((xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r), fill=(255, 245, 145), outline=(20, 20, 20))
        track_id = record.get("track_id", "unknown")
        draw.text((target[0] + 6, target[1] + 6), f"track {track_id}", fill=(255, 245, 145), font=font)


def _map_point(
    x: float,
    y: float,
    ref_w: int,
    ref_h: int,
    target: tuple[int, int, int, int],
) -> tuple[int, int]:
    target_w = max(1, target[2] - target[0])
    target_h = max(1, target[3] - target[1])
    return (
        int(target[0] + (x / max(1, ref_w - 1)) * target_w),
        int(target[1] + (y / max(1, ref_h - 1)) * target_h),
    )


def _fit_box(
    source_w: int,
    source_h: int,
    box: tuple[int, int, int, int],
    top_padding: int = 0,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    y0 += top_padding
    max_w = max(1, x1 - x0 - 16)
    max_h = max(1, y1 - y0 - 12)
    scale = min(max_w / max(1, source_w), max_h / max(1, source_h))
    width = max(1, int(source_w * scale))
    height = max(1, int(source_h * scale))
    px = x0 + (x1 - x0 - width) // 2
    py = y0 + (y1 - y0 - height) // 2
    return (px, py, px + width, py + height)


def _draw_center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.text((box[0] + 12, box[1] + 44), text, fill=(143, 153, 168), font=font)


def _retention_passed(retention: dict[str, Any] | None) -> bool:
    return bool(retention and retention.get("passed"))


def _artifact_label(
    silhouette_path: Path | None,
    depth_path: Path | None,
    skeleton_records: list[dict[str, Any]],
) -> str:
    labels = []
    if silhouette_path is not None:
        labels.append("mask")
    if depth_path is not None:
        labels.append("depth")
    if skeleton_records:
        labels.append("skeleton")
    return ",".join(labels) if labels else "none"


def _shape_from_skeletons(records: list[dict[str, Any]]) -> tuple[int, int] | None:
    xs: list[float] = []
    ys: list[float] = []
    for record in records:
        for point in record.get("keypoints", []):
            if isinstance(point, dict):
                xs.append(float(point.get("x", 0.0)))
                ys.append(float(point.get("y", 0.0)))
    if not xs or not ys:
        return None
    return (max(1, int(max(ys)) + 1), max(1, int(max(xs)) + 1))

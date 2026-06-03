from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable

import numpy as np


try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - depends on local environment
    cv2 = None

try:
    import imageio.v3 as iio  # type: ignore
except Exception:  # pragma: no cover - depends on local environment
    iio = None


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".ppm", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ms: float
    image: np.ndarray
    source_name: str


@dataclass(frozen=True)
class InputSummary:
    input_type: str
    source_count: int
    readable_frames: int
    skipped_frames: int

    def to_json(self) -> dict[str, int | str]:
        return {
            "input_type": self.input_type,
            "source_count": self.source_count,
            "readable_frames": self.readable_frames,
            "skipped_frames": self.skipped_frames,
        }


def load_frames(input_path: Path, max_frames: int | None = None) -> tuple[list[Frame], InputSummary]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"input path does not exist: {path}")

    if path.is_dir():
        candidates = sorted(
            child for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        )
        frames: list[Frame] = []
        skipped = 0
        for candidate in candidates:
            image = read_image(candidate)
            if image is None:
                skipped += 1
                continue
            frames.append(Frame(len(frames), float(len(frames)), image, candidate.name))
            if max_frames is not None and len(frames) >= max_frames:
                break
        return frames, InputSummary("directory", len(candidates), len(frames), skipped)

    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return load_video_frames(path, max_frames=max_frames)

    if suffix in IMAGE_EXTENSIONS:
        image = read_image(path)
        if image is None:
            return [], InputSummary("image", 1, 0, 1)
        return [Frame(0, 0.0, image, path.name)], InputSummary("image", 1, 1, 0)

    raise ValueError(f"unsupported input type for {path}")


def load_video_frames(path: Path, max_frames: int | None = None) -> tuple[list[Frame], InputSummary]:
    if cv2 is None:
        return load_video_frames_imageio(path, max_frames=max_frames)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return load_video_frames_imageio(path, max_frames=max_frames)

    frames: list[Frame] = []
    skipped = 0
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    while True:
        ok, image_bgr = capture.read()
        if not ok:
            break
        if image_bgr is None:
            skipped += 1
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or len(frames))
        frames.append(Frame(len(frames), timestamp_ms, image, path.name))
        if max_frames is not None and len(frames) >= max_frames:
            break
    capture.release()
    return frames, InputSummary("video", source_count or len(frames), len(frames), skipped)


def load_video_frames_imageio(path: Path, max_frames: int | None = None) -> tuple[list[Frame], InputSummary]:
    if iio is None:
        raise RuntimeError("video input requires OpenCV (cv2) or imageio[ffmpeg]")

    frames: list[Frame] = []
    skipped = 0
    try:
        properties = iio.improps(path)
        source_count = int(properties.shape[0]) if properties.shape and len(properties.shape) >= 4 else 0
    except Exception:
        source_count = 0

    try:
        iterator = iio.imiter(path)
        for raw_frame in iterator:
            if raw_frame is None:
                skipped += 1
                continue
            image = normalize_video_frame(np.asarray(raw_frame))
            if image is None:
                skipped += 1
                continue
            frames.append(Frame(len(frames), float(len(frames)), image, path.name))
            if max_frames is not None and len(frames) >= max_frames:
                break
    except Exception as exc:
        raise RuntimeError(
            "could not read video input with OpenCV or imageio[ffmpeg]; "
            "install ffmpeg support or provide an image frame directory"
        ) from exc

    return frames, InputSummary("video", source_count or len(frames), len(frames), skipped)


def normalize_video_frame(frame: np.ndarray) -> np.ndarray | None:
    if frame.ndim == 2:
        return frame.astype(np.uint8)
    if frame.ndim != 3:
        return None
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.shape[2] != 3:
        return None
    return frame.astype(np.uint8)


def read_image(path: Path) -> np.ndarray | None:
    if cv2 is not None:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image_bgr is not None:
            if image_bgr.ndim == 2:
                return image_bgr.astype(np.uint8)
            if image_bgr.shape[2] == 4:
                image_bgr = image_bgr[:, :, :3]
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

    if path.suffix.lower() in {".ppm", ".pgm"}:
        return read_portable_anymap(path)
    return None


def read_portable_anymap(path: Path) -> np.ndarray | None:
    data = path.read_bytes()
    tokens = list(islice(_pnm_tokens(data), 4))
    if len(tokens) < 4:
        return None

    magic = tokens[0]
    if magic not in {b"P5", b"P6"}:
        return None
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    if max_value <= 0 or max_value > 255:
        return None

    header_end = _pnm_header_end(data, 4)
    if header_end is None:
        return None
    channels = 3 if magic == b"P6" else 1
    expected = width * height * channels
    payload = data[header_end : header_end + expected]
    if len(payload) != expected:
        return None
    array = np.frombuffer(payload, dtype=np.uint8)
    if channels == 3:
        return array.reshape((height, width, 3))
    return array.reshape((height, width))


def _pnm_tokens(data: bytes) -> Iterable[bytes]:
    index = 0
    while index < len(data):
        while index < len(data) and chr(data[index]).isspace():
            index += 1
        if index < len(data) and data[index] == ord("#"):
            while index < len(data) and data[index] not in (10, 13):
                index += 1
            continue
        start = index
        while index < len(data) and not chr(data[index]).isspace():
            index += 1
        if start < index:
            yield data[start:index]


def _pnm_header_end(data: bytes, token_count: int) -> int | None:
    seen = 0
    index = 0
    while index < len(data):
        while index < len(data) and chr(data[index]).isspace():
            index += 1
        if index < len(data) and data[index] == ord("#"):
            while index < len(data) and data[index] not in (10, 13):
                index += 1
            continue
        while index < len(data) and not chr(data[index]).isspace():
            index += 1
        seen += 1
        if seen == token_count:
            while index < len(data) and chr(data[index]).isspace():
                index += 1
            return index
    return None

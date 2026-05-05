"""Per-video stable bbox via MediaPipe FaceLandmarker.

Uses 15 skull-anchored MediaPipe landmark indices (eye corners, nose,
temples) that do not move with mouth/blink, so the resulting bbox is
stable enough to share across all frames. Indices and size calibration
are ported from HRAvatar's preprocess/_smirk_constants.py to keep the
behaviour identical to the wider codebase the user already validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


STABLE_LANDMARK_INDICES = np.array(
    [33, 133, 362, 263, 1, 4, 5, 6, 168, 195, 197, 234, 454, 127, 356],
    dtype=np.int64,
)
STABLE_LANDMARK_SIZE_CALIBRATION = 1.55


def build_face_landmarker(
    model_asset_path: str,
    with_blendshapes: bool = False,
) -> mp_vision.FaceLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=model_asset_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        output_face_blendshapes=with_blendshapes,
        output_facial_transformation_matrixes=with_blendshapes,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def _build_face_landmarker(model_asset_path: str) -> mp_vision.FaceLandmarker:
    return build_face_landmarker(model_asset_path, with_blendshapes=False)


def _detect_landmarks_pixels(
    detector: mp_vision.FaceLandmarker,
    bgr: np.ndarray,
) -> Optional[np.ndarray]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    pts = np.array(
        [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]],
        dtype=np.float32,
    )
    return pts


def _bbox_from_stable_subset(
    pts_full: np.ndarray,
    bb_scale: float,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """Return [xmin, ymin, xmax, ymax] in pixel coords."""
    pts = pts_full[STABLE_LANDMARK_INDICES]
    x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.2 * y_max + 0.8 * y_min
    half = max(x_center - x_min, y_center - y_min)
    size = bb_scale * 2.0 * half * STABLE_LANDMARK_SIZE_CALIBRATION
    half_size = size / 2.0
    xmin = x_center - half_size
    xmax = x_center + half_size
    ymin = y_center - half_size
    ymax = y_center + half_size
    return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)


def _square_pad_to_image(
    bbox: np.ndarray, image_h: int, image_w: int
) -> np.ndarray:
    """Make sure the bbox stays inside the image without changing its size."""
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    side = max(w, h)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = side / 2.0
    if cx - half < 0:
        cx = half
    if cy - half < 0:
        cy = half
    if cx + half > image_w:
        cx = image_w - half
    if cy + half > image_h:
        cy = image_h - half
    return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def compute_stable_bbox(
    raw_dir: Path,
    model_asset_path: str,
    bb_scale: float = 2.0,
    per_frame: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MediaPipe over raw frames; return (per_frame_bbox, valid_mask).

    per_frame_bbox: (T, 4) float32 [xmin, ymin, xmax, ymax]. When per_frame is
        False (default) the same bbox is broadcast to every row.
    valid_mask: (T,) bool — False where MediaPipe failed to detect a face.

    Even when per_frame=False, every row is filled (median over valid frames),
    so callers can index uniformly.
    """
    paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() == ".png")
    if not paths:
        raise RuntimeError(f"No PNG frames under {raw_dir}")

    detector = _build_face_landmarker(model_asset_path)

    bboxes = []
    valid = []
    image_h = image_w = None
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            valid.append(False)
            bboxes.append(None)
            continue
        if image_h is None:
            image_h, image_w = bgr.shape[:2]
        pts = _detect_landmarks_pixels(detector, bgr)
        if pts is None:
            valid.append(False)
            bboxes.append(None)
            continue
        bbox = _bbox_from_stable_subset(pts, bb_scale, image_h, image_w)
        bboxes.append(bbox)
        valid.append(True)

    valid_arr = np.array(valid, dtype=bool)
    if not valid_arr.any():
        raise RuntimeError(
            "MediaPipe could not detect a face in any frame; "
            "check that the input video is a clear frontal recording."
        )

    valid_bboxes = np.stack([b for b in bboxes if b is not None], axis=0)

    T = len(paths)
    out = np.zeros((T, 4), dtype=np.float32)
    if per_frame:
        for i, b in enumerate(bboxes):
            if b is None:
                out[i] = np.median(valid_bboxes, axis=0)
            else:
                out[i] = b
        for i in range(T):
            out[i] = _square_pad_to_image(out[i], image_h, image_w)
    else:
        median_bbox = np.median(valid_bboxes, axis=0).astype(np.float32)
        median_bbox = _square_pad_to_image(median_bbox, image_h, image_w)
        out[:] = median_bbox

    return out, valid_arr


def crop_and_resize_frames(
    raw_dir: Path,
    out_dir: Path,
    bboxes: np.ndarray,
    out_size: int,
) -> list[Path]:
    """Crop each raw frame by its bbox and resize to (out_size, out_size).

    Returns the list of written paths (sorted by frame index).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() == ".png")
    if len(paths) != bboxes.shape[0]:
        raise RuntimeError(
            f"Frame count mismatch: {len(paths)} pngs vs {bboxes.shape[0]} bboxes"
        )

    written = []
    for idx, p in enumerate(paths):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        h, w = bgr.shape[:2]
        xmin, ymin, xmax, ymax = bboxes[idx]
        x0 = int(max(0, np.floor(xmin)))
        y0 = int(max(0, np.floor(ymin)))
        x1 = int(min(w, np.ceil(xmax)))
        y1 = int(min(h, np.ceil(ymax)))
        crop = bgr[y0:y1, x0:x1]
        if crop.shape[0] != crop.shape[1]:
            side = max(crop.shape[0], crop.shape[1])
            pad = np.zeros((side, side, 3), dtype=crop.dtype)
            pad[:crop.shape[0], :crop.shape[1]] = crop
            crop = pad
        out_img = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
        out_path = out_dir / f"{idx:05d}.png"
        cv2.imwrite(str(out_path), out_img)
        written.append(out_path)
    return written

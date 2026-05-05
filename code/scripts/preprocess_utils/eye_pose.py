"""Per-frame eye pose estimation from MediaPipe ARKit blendshapes.

Ported from MTamon/smirk@release/cuda128 utils/eye_pose.py — kept here as
a local copy so the Gaussian-HS preprocessor does not need to inject the
SMIRK repo onto sys.path.

Input: a directory of cropped frames (the same `image/` SMIRK was given
without --crop). For each frame we run a MediaPipe FaceLandmarker
configured for blendshapes and convert ARKit eye-look scores into the
12D rot6d ``eyes_pose`` and 2D ``eyelids`` that the downstream FLAME
parameter assembly expects.

Blendshape values are *crop-invariant* scalar scores: skipping SMIRK's
internal --crop and computing them on our own square crops keeps the
SMIRK encoder pose/cam in clean (S/224)-rescaled coordinates while still
giving us proper eye tracking.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from scripts.preprocess_utils.stable_bbox import build_face_landmarker


_MAX_PITCH = 0.6
_MAX_YAW = 0.6


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = axis_angle / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float32)
    return (
        np.eye(3, dtype=np.float32)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )


def _matrix_to_rotation_6d(mat: np.ndarray) -> np.ndarray:
    return mat[:, :2].T.reshape(-1).astype(np.float32)


def _eyes_pose_from_blendshapes(bs: dict) -> np.ndarray:
    """ARKit blendshape dict -> (12,) rot6d eyes_pose."""
    def g(k: str) -> float:
        return float(bs.get(k, 0.0))

    left_pitch = (g("eyeLookDownLeft") - g("eyeLookUpLeft")) * _MAX_PITCH
    left_yaw = (g("eyeLookInLeft") - g("eyeLookOutLeft")) * _MAX_YAW
    right_pitch = (g("eyeLookDownRight") - g("eyeLookUpRight")) * _MAX_PITCH
    right_yaw = (g("eyeLookOutRight") - g("eyeLookInRight")) * _MAX_YAW

    left_aa = np.array([left_pitch, left_yaw, 0.0], dtype=np.float32)
    right_aa = np.array([right_pitch, right_yaw, 0.0], dtype=np.float32)

    left_6d = _matrix_to_rotation_6d(_axis_angle_to_matrix(left_aa))
    right_6d = _matrix_to_rotation_6d(_axis_angle_to_matrix(right_aa))
    return np.concatenate([left_6d, right_6d], axis=0).astype(np.float32)


def _eyelids_from_blendshapes(bs: dict) -> np.ndarray:
    return np.array(
        [float(bs.get("eyeBlinkLeft", 0.0)), float(bs.get("eyeBlinkRight", 0.0))],
        dtype=np.float32,
    )


def extract_eyes_per_frame(
    image_dir: Path,
    model_asset_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (eyes_pose, eyelids, valid) for each frame in image_dir.

    eyes_pose: (T, 12) float32 — rot6d for [left|right] eyeball.
    eyelids:   (T,  2) float32 — [eyeBlinkLeft, eyeBlinkRight] in [0, 1].
    valid:     (T,)    bool    — False where MediaPipe failed to detect.
                                 Identity rot6d is written to invalid rows.
    """
    detector = build_face_landmarker(model_asset_path, with_blendshapes=True)

    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() == ".png")
    if not paths:
        raise RuntimeError(f"No PNG frames under {image_dir}")

    identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    identity_eyes = np.concatenate([identity_6d, identity_6d], axis=0)

    eyes = np.zeros((len(paths), 12), dtype=np.float32)
    lids = np.zeros((len(paths), 2), dtype=np.float32)
    valid = np.zeros(len(paths), dtype=bool)

    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            eyes[i] = identity_eyes
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if not result.face_blendshapes:
            eyes[i] = identity_eyes
            continue
        bs = {b.category_name: float(b.score) for b in result.face_blendshapes[0]}
        eyes[i] = _eyes_pose_from_blendshapes(bs)
        lids[i] = _eyelids_from_blendshapes(bs)
        valid[i] = True

    return eyes, lids, valid

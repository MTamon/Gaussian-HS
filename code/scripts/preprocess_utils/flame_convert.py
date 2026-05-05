"""SMIRK -> Gaussian-HS (IMavatar format) FLAME parameter conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


SMIRK_FOCAL_AT_224 = 5000.0


def rot6d_to_axis_angle(v: np.ndarray) -> np.ndarray:
    a1, a2 = v[:3].astype(np.float64), v[3:6].astype(np.float64)
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        return np.zeros(3, dtype=np.float32)
    b1 = a1 / n1
    b2 = a2 - (b1 @ a2) * b1
    n2 = np.linalg.norm(b2)
    if n2 < 1e-8:
        return np.zeros(3, dtype=np.float32)
    b2 = b2 / n2
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    aa, _ = cv2.Rodrigues(R)
    return aa.reshape(3).astype(np.float32)


def build_pose15(
    pose6: np.ndarray,
    eyes_pose12: Optional[np.ndarray] = None,
) -> np.ndarray:
    """[global(3) | neck(3) | jaw(3) | L_eye(3) | R_eye(3)] in axis-angle."""
    g = pose6[0:3].astype(np.float32)
    n = np.zeros(3, dtype=np.float32)
    j = pose6[3:6].astype(np.float32)
    if eyes_pose12 is not None:
        l = rot6d_to_axis_angle(eyes_pose12[0:6])
        r = rot6d_to_axis_angle(eyes_pose12[6:12])
    else:
        l = np.zeros(3, dtype=np.float32)
        r = np.zeros(3, dtype=np.float32)
    return np.concatenate([g, n, j, l, r]).astype(np.float32)


def smirk_cam_to_world_mat(
    cam: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """SMIRK pseudo-orthographic (s, tx, ty) -> 4x4 world_mat (+z forward).

    real_dataset.py:227-228 flips the z column / element after loading,
    so we write the un-flipped (+z forward) convention here.
    """
    s, tx_n, ty_n = float(cam[0]), float(cam[1]), float(cam[2])
    s = max(s, 1e-6)
    focal_pix_S = SMIRK_FOCAL_AT_224 * (image_size / 224.0)
    tz = focal_pix_S / (s * image_size / 2.0)
    tx = tx_n / s
    ty = -ty_n / s
    M = np.eye(4, dtype=np.float32)
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M


def build_intrinsics(image_size: int) -> list:
    """IMavatar normalized intrinsics list [fx_norm, fy_norm, cx_norm, cy_norm]."""
    focal_norm = SMIRK_FOCAL_AT_224 / 224.0
    return [float(focal_norm), float(focal_norm), 0.5, 0.5]


def interpolate_invalid_frames(
    arr: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Per-frame linear interpolation over time for rows where valid is False.

    arr: (T, D) float
    valid: (T,) bool
    Returns a copy with invalid rows filled by linear interpolation between
    surrounding valid rows. Edge invalid rows are nearest-filled.
    """
    out = arr.astype(np.float32).copy()
    T = out.shape[0]
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return out
    if len(valid_idx) == T:
        return out
    for d in range(out.shape[1]):
        out[:, d] = np.interp(np.arange(T), valid_idx, out[valid_idx, d])
    return out


def aggregate_shape(
    shape_smirk: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Per-subject shape: mean over valid frames, sliced to first 100 dims."""
    if valid.any():
        m = shape_smirk[valid].mean(axis=0)
    else:
        m = shape_smirk.mean(axis=0)
    return m[:100].astype(np.float32)


def assemble_flame_params(
    smirk_pt_path: Path,
    image_size: int,
    image_relpath_template: str = "image/{:05d}",
    eyes_pose_external: Optional[np.ndarray] = None,
    eyes_valid_external: Optional[np.ndarray] = None,
) -> dict:
    """Read SMIRK output .pt and build the IMavatar flame_params.json dict.

    eyes_pose_external / eyes_valid_external (both optional, both with leading
    dim T) override SMIRK's internal eyes_pose. Use this to feed eyes derived
    from a separate MediaPipe blendshape pass on the saved crops, since SMIRK
    --with_eye_pose forces an internal re-crop that breaks the cam→world_mat
    coordinate alignment.
    """
    blob = torch.load(str(smirk_pt_path), weights_only=False, map_location="cpu")

    shape = blob["shape"].numpy().astype(np.float32)
    exp = blob["exp"].numpy().astype(np.float32)
    pose = blob["pose"].numpy().astype(np.float32)
    cam = blob["cam"].numpy().astype(np.float32)
    valid = blob["valid_mask"].numpy().astype(bool)

    if eyes_pose_external is not None:
        eyes_pose = np.asarray(eyes_pose_external, dtype=np.float32)
        eyes_valid = (
            np.asarray(eyes_valid_external, dtype=bool)
            if eyes_valid_external is not None
            else np.ones(eyes_pose.shape[0], dtype=bool)
        )
    elif "eyes_pose" in blob:
        eyes_pose = blob["eyes_pose"].numpy().astype(np.float32)
        eyes_valid = valid.copy()
    else:
        eyes_pose = None
        eyes_valid = None

    T = shape.shape[0]
    if not (exp.shape[0] == pose.shape[0] == cam.shape[0] == valid.shape[0] == T):
        raise RuntimeError(
            f"SMIRK output dimension mismatch: shape={shape.shape}, exp={exp.shape}, "
            f"pose={pose.shape}, cam={cam.shape}, valid={valid.shape}"
        )
    if eyes_pose is not None and eyes_pose.shape[0] != T:
        raise RuntimeError(
            f"eyes_pose length mismatch: SMIRK T={T}, eyes_pose T={eyes_pose.shape[0]}"
        )

    exp_filled = interpolate_invalid_frames(exp, valid)
    pose_filled = interpolate_invalid_frames(pose, valid)
    cam_filled = interpolate_invalid_frames(cam, valid)
    if eyes_pose is not None:
        eyes_filled = interpolate_invalid_frames(eyes_pose, eyes_valid)
    else:
        eyes_filled = None

    shape100 = aggregate_shape(shape, valid)

    frames = []
    for t in range(T):
        pose15 = build_pose15(
            pose_filled[t],
            None if eyes_filled is None else eyes_filled[t],
        )
        world_mat = smirk_cam_to_world_mat(cam_filled[t], image_size)
        frames.append({
            "file_path": image_relpath_template.format(t),
            "world_mat": world_mat.tolist(),
            "expression": exp_filled[t, :50].astype(np.float32).tolist(),
            "pose": pose15.tolist(),
        })

    return {
        "intrinsics": build_intrinsics(image_size),
        "shape_params": shape100.tolist(),
        "frames": frames,
    }

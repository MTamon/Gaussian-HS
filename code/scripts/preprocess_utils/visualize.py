"""Black-background debug video for SMIRK preprocessing output.

Overlays:
  - 68 FLAME face landmarks (red) projected from the per-frame FLAME params
  - 4 DWpose shoulder/torso keypoints (cyan) and the OpenPose-style skeleton
  - Status text (frame index, smirk_valid, cam tuple)

Used as a sanity check that the SMIRK -> Gaussian-HS parameter conversion
landed the FLAME mesh on the right pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from datasets.real_dataset import draw_bodypose
from flame.FLAME import FLAME
from flame.lbs import vertices2landmarks


def _load_flame(flame_dir: Path, shape100: np.ndarray) -> FLAME:
    flame_model_path = flame_dir / "generic_model.pkl"
    lmk_embedding_path = flame_dir / "landmark_embedding.npy"
    if not flame_model_path.exists() or not lmk_embedding_path.exists():
        raise RuntimeError(
            f"FLAME assets missing under {flame_dir}: "
            f"need generic_model.pkl + landmark_embedding.npy"
        )
    shape_t = torch.from_numpy(shape100.astype(np.float32)).unsqueeze(0)
    canonical_expression = torch.zeros(1, 50, dtype=torch.float32)
    canonical_pose = torch.tensor(0.0, dtype=torch.float32)
    flame = FLAME(
        flame_model_path=str(flame_model_path),
        lmk_embedding_path=str(lmk_embedding_path),
        n_shape=100,
        n_exp=50,
        shape_params=shape_t,
        canonical_expression=canonical_expression,
        canonical_pose=canonical_pose,
    )
    flame = flame.cuda().eval()
    return flame


def _project_points_to_image(
    pts3d: np.ndarray,
    world_mat_4x4: np.ndarray,
    focal_pix: float,
    cx_pix: float,
    cy_pix: float,
) -> np.ndarray:
    """+z forward, identity rotation world_mat -> pixel coords."""
    R = world_mat_4x4[:3, :3]
    t = world_mat_4x4[:3, 3]
    cam = pts3d @ R.T + t
    z = np.maximum(cam[:, 2], 1e-6)
    u = focal_pix * cam[:, 0] / z + cx_pix
    v = focal_pix * cam[:, 1] / z + cy_pix
    return np.stack([u, v], axis=1)


def render_debug_video(
    flame_params: dict,
    image_dir: Path,
    dwpose_dir: Path,
    valid_mask: np.ndarray,
    cam_per_frame: np.ndarray,
    out_path: Path,
    flame_dir: Path,
    image_size: int,
    fps: float,
) -> None:
    """Write a debug.mp4 with FLAME + DWpose overlays on a black canvas."""
    shape100 = np.array(flame_params["shape_params"], dtype=np.float32)
    flame = _load_flame(flame_dir, shape100)

    intr = flame_params["intrinsics"]
    focal_pix = intr[0] * image_size
    cx_pix = intr[2] * image_size
    cy_pix = intr[3] * image_size

    frames = flame_params["frames"]
    T = len(frames)

    expressions = torch.from_numpy(
        np.stack([np.array(f["expression"], dtype=np.float32) for f in frames], axis=0)
    ).cuda()
    poses = torch.from_numpy(
        np.stack([np.array(f["pose"], dtype=np.float32) for f in frames], axis=0)
    ).cuda()

    canvases = []
    batch_size = 32
    for batch_start in range(0, T, batch_size):
        batch_end = min(batch_start + batch_size, T)
        with torch.no_grad():
            verts, _, _ = flame(
                expressions[batch_start:batch_end],
                poses[batch_start:batch_end],
            )
            lmks2d, _ = flame.find_landmarks(verts, poses[batch_start:batch_end])
        lmks_np = lmks2d.cpu().numpy().astype(np.float32)

        for i, t in enumerate(range(batch_start, batch_end)):
            canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            world_mat = np.array(frames[t]["world_mat"], dtype=np.float32)
            uv = _project_points_to_image(
                lmks_np[i], world_mat, focal_pix, cx_pix, cy_pix,
            )
            for (u, v) in uv:
                ui, vi = int(round(u)), int(round(v))
                if 0 <= ui < image_size and 0 <= vi < image_size:
                    cv2.circle(canvas, (ui, vi), 2, (0, 0, 255), -1)

            dwpose_path = dwpose_dir / f"{t:05d}.npy"
            if dwpose_path.exists():
                dw = np.load(str(dwpose_path), allow_pickle=True).item()
                candidate = np.asarray(dw["bodies"]["candidate"], dtype=np.float32)
                subset = np.asarray(dw["bodies"]["subset"], dtype=np.float32)
                canvas = draw_bodypose(canvas, candidate, subset, point_size=4, line_size=4)
                shoulder_idx = [0, 1, 2, 5]
                if candidate.shape[0] > max(shoulder_idx):
                    for k in shoulder_idx:
                        x, y = candidate[k, 0], candidate[k, 1]
                        if x >= 0 and y >= 0:
                            ui = int(round(float(x) * image_size))
                            vi = int(round(float(y) * image_size))
                            if 0 <= ui < image_size and 0 <= vi < image_size:
                                cv2.circle(canvas, (ui, vi), 5, (255, 255, 0), -1)

            cam_t = cam_per_frame[t] if cam_per_frame is not None else None
            cam_str = (
                f"cam=({cam_t[0]:.2f},{cam_t[1]:+.2f},{cam_t[2]:+.2f})"
                if cam_t is not None else "cam=N/A"
            )
            valid_str = f"valid={int(valid_mask[t])}" if valid_mask is not None else ""
            text = f"#{t:05d} {valid_str} {cam_str}"
            cv2.putText(
                canvas, text, (8, image_size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

            canvases.append(canvas)

    frames_tensor = torch.from_numpy(np.stack(canvases, axis=0))
    frames_tensor = frames_tensor[..., [2, 1, 0]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torchvision.io as tvio
    tvio.write_video(
        str(out_path),
        frames_tensor,
        fps=float(fps),
        video_codec="h264",
    )

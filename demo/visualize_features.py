#!/usr/bin/env python3
"""Overlay prepared Gaussian-HS dataset features (mask / DWpose / FLAME 2D
landmarks) on the source RGB frames so the user can visually confirm
preprocessing alignment before training. See demo/README.md for usage."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def _resolve_repo_paths(here: Path):
    repo = here.parent
    code = repo / "code"
    if not (code / "datasets" / "real_dataset.py").is_file():
        raise SystemExit(
            f"[visualize_features] cannot find {code}/datasets/real_dataset.py"
        )
    return repo, code


def _load_drawing_helpers(code_dir: Path):
    sys.path.insert(0, str(code_dir))
    from datasets import real_dataset
    return real_dataset.draw_bodypose, real_dataset.draw_handpose


def _resolve_data_root(repo: Path, override):
    if override:
        return Path(override).resolve()
    return (repo / ".." / "data" / "datasets").resolve()


def _detect_flame_coord_mode(keypoints):
    arr = np.asarray(keypoints, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return "pixel"
    return "pixel" if np.max(np.abs(arr)) > 2.0 else "ndc"


def _project_flame_keypoints(keypoints, h, w, mode):
    arr = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    if mode == "ndc":
        x = (arr[:, 0] + 1.0) * 0.5 * w
        y = (1.0 - (arr[:, 1] + 1.0) * 0.5) * h
        arr = np.stack([x, y], axis=-1)
    return arr.astype(np.int32)


def _build_dwpose_layer(dwpose, h, w, draw_bodypose, draw_handpose):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    bodies = dwpose.get("bodies", {})
    candidate = bodies.get("candidate")
    subset = bodies.get("subset")
    if candidate is not None and subset is not None and len(candidate) > 0:
        canvas = draw_bodypose(canvas, candidate, subset)
    hands = dwpose.get("hands")
    if hands is not None and len(hands) > 0:
        canvas = draw_handpose(canvas, hands)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _composite(image, mask, dwpose_layer, flame_pts, frame_idx, file_path):
    h, w = image.shape[:2]
    base = image.astype(np.float32)

    if mask is not None:
        green = np.zeros_like(base)
        green[..., 1] = 255.0
        m = (mask > 0.5).astype(np.float32)[..., None]
        base = base * (1.0 - m * 0.25) + green * (m * 0.25)

    if dwpose_layer is not None:
        nz = (dwpose_layer.sum(axis=-1) > 0).astype(np.float32)[..., None]
        base = base * (1.0 - nz * 0.85) + dwpose_layer.astype(np.float32) * (nz * 0.85)

    base = base.clip(0.0, 255.0).astype(np.uint8)

    if flame_pts is not None:
        for x, y in flame_pts:
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(base, (int(x), int(y)), 2, (255, 32, 32), thickness=-1)

    label = f"#{frame_idx:05d}  {file_path}"
    text_w = min(len(label) * 9 + 12, w)
    cv2.rectangle(base, (0, h - 22), (text_w, h), (0, 0, 0), thickness=-1)
    cv2.putText(base, label, (6, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Overlay-visualize prepared Gaussian-HS dataset features."
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="dataset root (default: <repo>/../data/datasets)")
    parser.add_argument("--subject", type=str, default="001")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--num-frames", type=int, default=60,
                        help="render at most N frames after striding (0 = all)")
    parser.add_argument("--stride", type=int, default=1,
                        help="sample every Nth frame")
    parser.add_argument("--mp4", action="store_true",
                        help="also compose overlay.mp4 (10 fps, h264)")
    parser.add_argument("--out", type=str, default=None,
                        help="output dir (default demo/output/01_overlay/<subject>/<split>)")
    parser.add_argument("--flame-coords",
                        choices=["pixel", "ndc", "auto", "none"], default="auto",
                        help="how to interpret flame_keypoints coordinates")
    parser.add_argument("--json-name", type=str, default="flame_params.json")
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    repo, code = _resolve_repo_paths(here)
    draw_bodypose, draw_handpose = _load_drawing_helpers(code)

    data_root = _resolve_data_root(repo, args.data_root)
    instance_dir = data_root / args.subject / args.subject / args.split
    if not instance_dir.is_dir():
        raise SystemExit(
            f"[visualize_features] dataset dir not found: {instance_dir}\n"
            "  Expected layout: <data-root>/<subject>/<subject>/<split>/"
            "{flame_params.json,image,mask,dwpose}\n"
            "  Run: bash download_assets.sh --flame_user USER --flame_pass PASS"
        )

    json_path = instance_dir / args.json_name
    if not json_path.is_file():
        raise SystemExit(f"[visualize_features] {json_path} missing")

    with open(json_path) as f:
        camera_dict = json.load(f)
    frames = camera_dict.get("frames", [])
    if not frames:
        raise SystemExit(f"[visualize_features] no frames in {json_path}")

    idxs = list(range(0, len(frames), max(args.stride, 1)))
    if args.num_frames > 0:
        idxs = idxs[: args.num_frames]
    if not idxs:
        raise SystemExit("[visualize_features] no frames selected")

    out_dir = (Path(args.out).resolve() if args.out
               else here / "output" / "01_overlay" / args.subject / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    flame_mode = args.flame_coords
    rendered = []

    for i, fidx in enumerate(idxs):
        frame = frames[fidx]
        file_path = frame["file_path"]
        img_name = Path(file_path).name
        image_path = instance_dir / f"{file_path}.png"

        if not image_path.is_file():
            print(f"[skip] missing image {image_path}")
            continue

        image = imageio.imread(image_path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        h, w = image.shape[:2]

        mask_path = instance_dir / "mask" / f"{img_name}.png"
        mask = None
        if mask_path.is_file():
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_raw is not None:
                mask = mask_raw.astype(np.float32) / 255.0
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        dwpose_path = instance_dir / "dwpose" / f"{img_name}.npy"
        dwpose_layer = None
        if dwpose_path.is_file():
            try:
                dwpose = np.load(str(dwpose_path), allow_pickle=True).item()
                dwpose_layer = _build_dwpose_layer(dwpose, h, w,
                                                   draw_bodypose, draw_handpose)
            except Exception as exc:
                print(f"[warn] failed to load DWpose at {dwpose_path}: {exc}")

        flame_pts = None
        if args.flame_coords != "none" and "flame_keypoints" in frame:
            kps = frame["flame_keypoints"]
            mode = (flame_mode if flame_mode in ("pixel", "ndc")
                    else _detect_flame_coord_mode(kps))
            flame_pts = _project_flame_keypoints(kps, h, w, mode)
            if flame_mode == "auto":
                flame_mode = mode

        composite = _composite(image, mask, dwpose_layer, flame_pts, i, file_path)
        out_path = out_dir / f"{i:05d}.png"
        imageio.imwrite(out_path, composite)
        rendered.append(out_path)

        if (i + 1) % 10 == 0 or i + 1 == len(idxs):
            print(f"[viz] {i + 1}/{len(idxs)} frames -> {out_dir}")

    if args.mp4 and rendered:
        try:
            import torch
            import torchvision.io as tvio
            stack = np.stack([imageio.imread(p) for p in rendered], axis=0)
            tvio.write_video(str(out_dir / "overlay.mp4"),
                             torch.from_numpy(stack).to(torch.uint8),
                             fps=10, video_codec="h264")
            print(f"[viz] wrote {out_dir / 'overlay.mp4'}")
        except Exception as exc:
            print(f"[warn] --mp4 requested but composition failed: {exc}")

    print(f"[viz] done. {len(rendered)} composite frame(s) under {out_dir}")


if __name__ == "__main__":
    main()

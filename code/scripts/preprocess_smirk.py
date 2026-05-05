"""SMIRK + RVM + face-parsing + DWpose video preprocessor for Gaussian-HS.

Produces an IMavatar-format dataset directory ready for `exp_runner.py`:

    <data-root>/<subject>/<subject>/<split>/
        image/00000.png ...        (S x S crops)
        mask/00000.png  ...        (RVM alpha)
        semantic/00000.png ...     (face-parsing.PyTorch labels)
        dwpose/00000.npy ...       (controlnet_aux DWPose, normalized xy)
        flame_params.json          (shape100 + per-frame expression50/pose15/world_mat)
        debug.mp4                  (black-bg overlay video)

Stages can be re-run individually via --start-stage / --end-stage (1..9).

Run from the repo's `code/` directory so that `from datasets.real_dataset
import draw_bodypose` and `from flame.FLAME import FLAME` resolve:

    cd code
    python scripts/preprocess_smirk.py --video ../path/to/input.mp4 \\
        --subject 999 --split train --image-size 512 --fps 25 \\
        --smirk-repo /home/mikawa/lab/outcome/smirk \\
        --hr-preprocess /home/mikawa/lab/outcome/HRAvatar/preprocess
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = _SCRIPT_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from scripts.preprocess_utils import flame_convert
from scripts.preprocess_utils import stable_bbox
from scripts.preprocess_utils import visualize
from scripts.preprocess_utils import eye_pose


print = partial(print, flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--subject", required=True, type=str)
    p.add_argument("--split", default="train", type=str)
    p.add_argument("--data-root", default=None, type=Path,
                   help="dataset root; defaults to <repo>/data/datasets")
    p.add_argument("--image-size", default=512, type=int)
    p.add_argument("--fps", default=25, type=int)
    p.add_argument("--bb-scale", default=2.0, type=float,
                   help="stable bbox scale; ~2.0 includes face + shoulders")
    p.add_argument("--per-frame-bbox", action="store_true",
                   help="use per-frame bbox (default: one fixed bbox per video)")
    p.add_argument("--smirk-repo", required=True, type=Path,
                   help="path to MTamon/smirk checkout")
    p.add_argument("--smirk-ckpt", default=None, type=Path,
                   help="path to SMIRK_em1.pt; defaults to <smirk-repo>/pretrained_models/SMIRK_em1.pt")
    p.add_argument("--hr-preprocess", required=True, type=Path,
                   help="path to HRAvatar/preprocess (for RVM + face-parsing submodules)")
    p.add_argument("--mediapipe-asset", default=None, type=Path,
                   help="path to face_landmarker.task; auto-discovered under HRAvatar/assets if omitted")
    p.add_argument("--smirk-batch-size", default=16, type=int)
    p.add_argument("--start-stage", default=1, type=int, choices=range(1, 10))
    p.add_argument("--end-stage", default=9, type=int, choices=range(1, 10))
    p.add_argument("--no-eye-pose", action="store_true",
                   help="skip MediaPipe-blendshape eye pose extraction (eyes left as zeros)")
    p.add_argument("--keep-work", action="store_true",
                   help="keep _work intermediates (default: keep, no-op for now)")
    p.add_argument("--python-bin", default=sys.executable, type=str,
                   help="python executable used for subprocess calls (SMIRK/RVM/face-parsing)")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> dict:
    repo_root = _CODE_DIR.parent
    data_root = args.data_root if args.data_root else repo_root / "data" / "datasets"
    out_root = data_root / args.subject / args.subject / args.split
    work = out_root / "_work"

    smirk_ckpt = args.smirk_ckpt
    if smirk_ckpt is None:
        smirk_ckpt = args.smirk_repo / "pretrained_models" / "SMIRK_em1.pt"

    if args.mediapipe_asset is not None:
        mp_asset = args.mediapipe_asset
    else:
        candidates = [
            args.hr_preprocess.parent / "assets" / "smirk" / "face_landmarker.task",
            args.hr_preprocess.parent / "assets" / "face_landmarker.task",
            args.smirk_repo / "assets" / "face_landmarker.task",
        ]
        mp_asset = next((c for c in candidates if c.exists()), None)
        if mp_asset is None:
            raise RuntimeError(
                "MediaPipe face_landmarker.task not found in any known location; "
                "pass --mediapipe-asset explicitly. Tried: "
                + ", ".join(str(c) for c in candidates)
            )

    flame_dir = _CODE_DIR / "flame" / "FLAME2020"

    return {
        "repo_root": repo_root,
        "out_root": out_root,
        "work": work,
        "raw_dir": work / "raw",
        "image_dir": out_root / "image",
        "mask_dir": out_root / "mask",
        "semantic_dir": out_root / "semantic",
        "dwpose_dir": out_root / "dwpose",
        "smirk_pt": work / "smirk_flame.pt",
        "bbox_npy": work / "bbox.npy",
        "valid_npy": work / "bbox_valid.npy",
        "cropped_mp4": work / "cropped.mp4",
        "eyes_npz": work / "eyes_pose.npz",
        "flame_json": out_root / "flame_params.json",
        "debug_mp4": out_root / "debug.mp4",
        "smirk_ckpt": smirk_ckpt,
        "mp_asset": mp_asset,
        "flame_dir": flame_dir,
    }


def _stage(name: str, n: int, start: int, end: int) -> bool:
    active = start <= n <= end
    if active:
        print(f"\n========== [Stage {n}] {name} ==========")
    else:
        print(f"\n[Stage {n}] {name} -- skipped")
    return active


def stage_extract_frames(args, paths) -> None:
    if not args.video.exists():
        raise RuntimeError(f"Input video not found: {args.video}")
    paths["raw_dir"].mkdir(parents=True, exist_ok=True)
    for old in paths["raw_dir"].glob("*.png"):
        old.unlink()
    S = args.image_size
    vf = (
        f"scale='if(gt(iw,ih),-2,{S})':'if(gt(iw,ih),{S},-2)',"
        f"fps={args.fps}"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(args.video),
        "-vf", vf,
        "-start_number", "0", "-q:v", "1",
        str(paths["raw_dir"] / "%05d.png"),
    ]
    print("ffmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def stage_stable_bbox(args, paths) -> None:
    bboxes, valid = stable_bbox.compute_stable_bbox(
        raw_dir=paths["raw_dir"],
        model_asset_path=str(paths["mp_asset"]),
        bb_scale=args.bb_scale,
        per_frame=args.per_frame_bbox,
    )
    paths["work"].mkdir(parents=True, exist_ok=True)
    np.save(str(paths["bbox_npy"]), bboxes)
    np.save(str(paths["valid_npy"]), valid)
    print(f"bbox saved: {paths['bbox_npy']} shape={bboxes.shape}, "
          f"valid={int(valid.sum())}/{len(valid)}")


def stage_crop(args, paths) -> None:
    bboxes = np.load(str(paths["bbox_npy"]))
    paths["image_dir"].mkdir(parents=True, exist_ok=True)
    for old in paths["image_dir"].glob("*.png"):
        old.unlink()
    written = stable_bbox.crop_and_resize_frames(
        raw_dir=paths["raw_dir"],
        out_dir=paths["image_dir"],
        bboxes=bboxes,
        out_size=args.image_size,
    )
    print(f"cropped: {len(written)} frames -> {paths['image_dir']}")

    if paths["cropped_mp4"].exists():
        paths["cropped_mp4"].unlink()
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(paths["image_dir"] / "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0",
        str(paths["cropped_mp4"]),
    ]
    print("ffmpeg (mp4):", " ".join(cmd))
    subprocess.run(cmd, check=True)


def stage_smirk(args, paths) -> None:
    if not paths["smirk_ckpt"].exists():
        raise RuntimeError(
            f"SMIRK checkpoint not found: {paths['smirk_ckpt']}. "
            "Download from MTamon/smirk releases or pass --smirk-ckpt."
        )
    if paths["smirk_pt"].exists():
        paths["smirk_pt"].unlink()
    cmd = [
        args.python_bin,
        "demos/demo_save_flame.py",
        "--input_path", str(paths["cropped_mp4"]),
        "--checkpoint", str(paths["smirk_ckpt"]),
        "--out_path", str(paths["smirk_pt"]),
        "--device", "cuda",
        "--batch_size", str(args.smirk_batch_size),
    ]
    print("smirk:", " ".join(cmd), f"(cwd={args.smirk_repo})")
    subprocess.run(cmd, check=True, cwd=str(args.smirk_repo))


def stage_rvm(args, paths) -> None:
    paths["mask_dir"].mkdir(parents=True, exist_ok=True)
    for old in paths["mask_dir"].glob("*.png"):
        old.unlink()
    rvm_ckpt = args.hr_preprocess / "submodules" / "RobustVideoMatting" / "rvm_resnet50.pth"
    if not rvm_ckpt.exists():
        raise RuntimeError(
            f"RVM checkpoint not found: {rvm_ckpt}. "
            "Run HRAvatar/download_assets.sh to fetch it."
        )
    cmd = [
        args.python_bin,
        "preprocess/submodules/RobustVideoMatting/inference.py",
        "--variant", "resnet50",
        "--checkpoint", str(rvm_ckpt),
        "--device", "cuda:0",
        "--input-source", str(paths["image_dir"]),
        "--output-alpha", str(paths["mask_dir"]),
        "--output-type", "png_sequence",
    ]
    cwd = args.hr_preprocess.parent
    print("rvm:", " ".join(cmd), f"(cwd={cwd})")
    subprocess.run(cmd, check=True, cwd=str(cwd))
    _rename_to_match_images(paths["image_dir"], paths["mask_dir"])


def _rename_to_match_images(image_dir: Path, mask_dir: Path) -> None:
    images = sorted(image_dir.glob("*.png"))
    masks = sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))
    if len(images) != len(masks):
        raise RuntimeError(
            f"image/mask count mismatch: {len(images)} vs {len(masks)} under {mask_dir}"
        )
    if all(img.stem == m.stem for img, m in zip(images, masks)):
        return
    tmp_dir = mask_dir.parent / (mask_dir.name + "_renaming_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for img, m in zip(images, masks):
        new_path = tmp_dir / f"{img.stem}{m.suffix}"
        shutil.move(str(m), str(new_path))
    for f in tmp_dir.iterdir():
        shutil.move(str(f), str(mask_dir / f.name))
    shutil.rmtree(tmp_dir)


def stage_face_parsing(args, paths) -> None:
    paths["semantic_dir"].mkdir(parents=True, exist_ok=True)
    for old in paths["semantic_dir"].glob("*.png"):
        old.unlink()
    parsing_ckpt = args.hr_preprocess / "submodules" / "face-parsing.PyTorch" / "res" / "cp" / "79999_iter.pth"
    if not parsing_ckpt.exists():
        raise RuntimeError(
            f"face-parsing checkpoint not found: {parsing_ckpt}. "
            "Run HRAvatar/download_assets.sh to fetch it."
        )
    cmd = [
        args.python_bin,
        "preprocess/submodules/face-parsing.PyTorch/test.py",
        "--dspth", str(paths["image_dir"]),
        "--respth", str(paths["semantic_dir"]),
    ]
    cwd = args.hr_preprocess.parent
    print("face-parsing:", " ".join(cmd), f"(cwd={cwd})")
    subprocess.run(cmd, check=True, cwd=str(cwd))

    if args.image_size != 512:
        import cv2
        for p in sorted(paths["semantic_dir"].glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape[0] == args.image_size:
                continue
            img_r = cv2.resize(
                img, (args.image_size, args.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imwrite(str(p), img_r)


def stage_dwpose(args, paths) -> None:
    """DWpose (yolox + dw-ll_ucoco_384) -> IMavatar-format .npy per frame.

    controlnet_aux's DWposeDetector.__call__ returns a drawn image, not the
    keypoint dict; we therefore call its underlying `pose_estimation` and
    rebuild the dict ourselves, matching the on-disk layout the loader at
    code/datasets/real_dataset.py:126-159 expects:

        {
          "bodies": {"candidate": (18, 2), "subset": (1, 18)},
          "hands":  (2, 21, 2),
          "faces":  (1, 68, 2),
        }
    All xy values are normalized to [0, 1] (real_dataset.py:516-518 multiplies
    by W/H when drawing). -1 marks an invisible / undetected keypoint.
    """
    paths["dwpose_dir"].mkdir(parents=True, exist_ok=True)
    for old in paths["dwpose_dir"].glob("*.npy"):
        old.unlink()

    import cv2
    from controlnet_aux.dwpose import DWposeDetector

    print("loading DWposeDetector (downloads yolox + dw-ll_ucoco_384 on first run)...")
    det = DWposeDetector(device="cuda")

    paths_in = sorted(paths["image_dir"].glob("*.png"))
    print(f"running DWpose on {len(paths_in)} frames...")
    with torch.no_grad():
        for p in paths_in:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            H, W = bgr.shape[:2]
            candidate, subset = det.pose_estimation(bgr)
            if candidate.shape[0] == 0:
                body = -np.ones((18, 2), dtype=np.float32)
                body_subset = -np.ones((1, 18), dtype=np.float32)
                hands = -np.ones((2, 21, 2), dtype=np.float32)
                faces = -np.ones((1, 68, 2), dtype=np.float32)
            else:
                candidate = candidate.astype(np.float32)
                subset = subset.astype(np.float32)
                candidate[..., 0] /= float(W)
                candidate[..., 1] /= float(H)
                body_score = subset[:1, :18].copy()
                for j in range(18):
                    body_score[0, j] = j if body_score[0, j] > 0.3 else -1
                un_visible = subset[:1, :] < 0.3
                cand_clean = candidate[:1].copy()
                cand_clean[un_visible] = -1
                body = cand_clean[0, :18].astype(np.float32)
                body_subset = body_score.astype(np.float32)
                hands_part = cand_clean[0, 92:113]
                hands_full = cand_clean[0, 113:]
                hands = np.stack([hands_part, hands_full], axis=0).astype(np.float32)
                faces = cand_clean[0, 24:92][None, ...].astype(np.float32)

            out = {
                "bodies": {"candidate": body, "subset": body_subset},
                "hands": hands,
                "faces": faces,
            }
            out_path = paths["dwpose_dir"] / f"{p.stem}.npy"
            np.save(str(out_path), out, allow_pickle=True)

    print(f"dwpose saved: {len(paths_in)} npy under {paths['dwpose_dir']}")


def stage_assemble_json(args, paths) -> None:
    eyes = lids = eyes_valid = None
    if not args.no_eye_pose:
        print("extracting eye pose from MediaPipe blendshapes on cropped frames...")
        eyes, lids, eyes_valid = eye_pose.extract_eyes_per_frame(
            image_dir=paths["image_dir"],
            model_asset_path=str(paths["mp_asset"]),
        )
        np.savez(str(paths["eyes_npz"]), eyes_pose=eyes, eyelids=lids, valid=eyes_valid)
        print(f"eye_pose: {paths['eyes_npz']} "
              f"valid={int(eyes_valid.sum())}/{len(eyes_valid)}")

    flame_dict = flame_convert.assemble_flame_params(
        smirk_pt_path=paths["smirk_pt"],
        image_size=args.image_size,
        eyes_pose_external=eyes,
        eyes_valid_external=eyes_valid,
    )
    paths["flame_json"].parent.mkdir(parents=True, exist_ok=True)
    with paths["flame_json"].open("w") as f:
        json.dump(flame_dict, f)
    print(f"flame_params.json: {paths['flame_json']} "
          f"(shape={len(flame_dict['shape_params'])}, frames={len(flame_dict['frames'])}, "
          f"intrinsics={flame_dict['intrinsics']})")


def stage_visualize(args, paths) -> None:
    with paths["flame_json"].open("r") as f:
        flame_dict = json.load(f)
    blob = torch.load(str(paths["smirk_pt"]), weights_only=False, map_location="cpu")
    valid = blob["valid_mask"].numpy().astype(bool)
    cam = blob["cam"].numpy().astype(np.float32)
    visualize.render_debug_video(
        flame_params=flame_dict,
        image_dir=paths["image_dir"],
        dwpose_dir=paths["dwpose_dir"],
        valid_mask=valid,
        cam_per_frame=cam,
        out_path=paths["debug_mp4"],
        flame_dir=paths["flame_dir"],
        image_size=args.image_size,
        fps=args.fps,
    )
    print(f"debug video: {paths['debug_mp4']}")


STAGES = [
    ("extract_frames", stage_extract_frames),
    ("stable_bbox", stage_stable_bbox),
    ("crop", stage_crop),
    ("smirk", stage_smirk),
    ("rvm", stage_rvm),
    ("face_parsing", stage_face_parsing),
    ("dwpose", stage_dwpose),
    ("assemble_json", stage_assemble_json),
    ("visualize", stage_visualize),
]


def main():
    args = _parse_args()
    paths = _resolve_paths(args)
    paths["out_root"].mkdir(parents=True, exist_ok=True)
    paths["work"].mkdir(parents=True, exist_ok=True)

    print(f"input video : {args.video}")
    print(f"output root : {paths['out_root']}")
    print(f"image size  : {args.image_size}")
    print(f"fps         : {args.fps}")
    print(f"stages      : {args.start_stage}..{args.end_stage}")

    for n, (name, fn) in enumerate(STAGES, start=1):
        if not _stage(name, n, args.start_stage, args.end_stage):
            continue
        fn(args, paths)

    print("\n[done]")


if __name__ == "__main__":
    main()

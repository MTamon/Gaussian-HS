# demo/

[日本語版](README.ja.md)

End-to-end demo scripts for Gaussian-HS. See the top-level `README.md` and
`CLAUDE.md` for project context.

The pipeline is four steps: build a per-subject dataset from a raw video,
inspect the alignment, train a personal model, then re-enact it from another
subject.

```
mp4  ──▶  00_preprocess_video.sh  ──▶  data/datasets/<subj>/...
                                          │
                                          ├──▶  01_preprocess_and_visualize.sh   (sanity check)
                                          ├──▶  02_train_subject.sh              (training)
                                          └──▶  03_cross_reenact.sh              (re-enactment)
```

Each script supports `--help` for the full flag list. Activate the env first:
`conda activate gaussian-hs`.

## 0. Build dataset from raw video (SMIRK + RVM + DWpose)

```sh
bash demo/00_preprocess_video.sh --video clip.mp4 --subject 999            # train split, 25 fps, 512px
bash demo/00_preprocess_video.sh --video clip.mp4 --subject 999 --split test
bash demo/00_preprocess_video.sh --video clip.mp4 --subject 999 --start-stage 7 --end-stage 9
```

Wraps `code/scripts/preprocess_smirk.py`. Produces an IMavatar-format
directory ready for steps 1–3:

```
data/datasets/<subject>/<subject>/<split>/
    image/00000.png ...        # square crops (S × S)
    mask/00000.png  ...        # RVM alpha matte
    semantic/00000.png ...     # face-parsing (CelebAMask-HQ labels)
    dwpose/00000.npy ...       # DWpose keypoints (IMavatar dict format)
    flame_params.json          # shape100 + per-frame expression50/pose15/world_mat
    debug.mp4                  # black-bg sanity overlay (see below)
    _work/                     # intermediates (raw frames, bbox, smirk_flame.pt, eyes_pose.npz)
```

The 9 stages can be re-run individually via `--start-stage N --end-stage M`
(useful when iterating on a single component):

| # | Stage          | Tool                      | Output |
|---|----------------|---------------------------|--------|
| 1 | extract_frames | ffmpeg                    | `_work/raw/*.png` |
| 2 | stable_bbox    | MediaPipe FaceLandmarker  | `_work/bbox.npy` |
| 3 | crop           | OpenCV + ffmpeg           | `image/`, `_work/cropped.mp4` |
| 4 | smirk          | MTamon/smirk subprocess   | `_work/smirk_flame.pt` |
| 5 | rvm            | RobustVideoMatting        | `mask/` |
| 6 | face_parsing   | face-parsing.PyTorch      | `semantic/` |
| 7 | dwpose         | controlnet_aux + mmpose   | `dwpose/` |
| 8 | assemble_json  | flame_convert + eye_pose  | `flame_params.json`, `_work/eyes_pose.npz` |
| 9 | visualize      | FLAME + draw_bodypose     | `debug.mp4` |

### External assets

The script invokes existing third-party stacks rather than re-implementing
them. Place / clone them at these defaults (or override with the matching
flag):

- `--smirk-repo` `/home/mikawa/lab/outcome/smirk` —
  [`MTamon/smirk@release/cuda128`](https://github.com/MTamon/smirk/tree/release/cuda128).
  Needs weights at `pretrained_models/SMIRK_em1.pt`.
- `--hr-preprocess` `/home/mikawa/lab/outcome/HRAvatar/preprocess` —
  HRAvatar's `submodules/RobustVideoMatting/` (`rvm_resnet50.pth`) and
  `submodules/face-parsing.PyTorch/res/cp/` (`79999_iter.pth`). Run
  `bash download_assets.sh` inside HRAvatar once to fetch them.
- DWpose weights (yolox + dw-ll_ucoco_384) auto-download to
  `~/.cache/torch/hub/checkpoints/` on first run via `controlnet_aux`.

### How FLAME parameters are produced

| Parameter      | Source                                                           |
|----------------|------------------------------------------------------------------|
| `shape_params` (100) | SMIRK shape, mean over valid frames, sliced to first 100 dims |
| `expression`   (50)  | SMIRK exp, as-is, linearly interpolated over invalid frames |
| `pose[0:3]`    (global) | SMIRK pose                                            |
| `pose[3:6]`    (neck)   | zeros (SMIRK does not estimate neck)                  |
| `pose[6:9]`    (jaw)    | SMIRK pose                                            |
| `pose[9:15]`   (L/R eye)| MediaPipe ARKit blendshapes (`eyeLook*`/`eyeBlink*`) → rot6d → axis-angle, computed independently from SMIRK so its `--with_eye_pose` internal re-crop does not perturb the cam coordinates. Pass `--no-eye-pose` to fall back to zeros. |
| `world_mat`    (4×4)    | DECA-style pseudo-perspective from SMIRK ortho cam (`s, tx, ty`) at `focal_pix = 5000 · S / 224`. Identity rotation, per-frame translation. |
| `intrinsics`   (4)      | Normalized pinhole `[5000/224, 5000/224, 0.5, 0.5]` regardless of S |

`world_mat` and `flame_pose` are refined per frame at training time via
`optimize_camera=True` / `optimize_pose=True`, so the initial values only
need to be approximately correct.

### Debug video sanity check

`debug.mp4` is the most efficient way to spot silent bugs. Inspect once per
new subject:

- **Red dots (FLAME 68 landmarks)** — should sit on the face. Off-face dots
  mean `world_mat` ↔ image alignment is wrong.
- **Cyan circles (DWpose nose / neck / shoulders)** — should sit on the
  expected body parts. Off-body points mean DWpose detection failed or the
  bbox crop hides the shoulders.
- **Status text (frame index, smirk_valid, cam tuple)** — `valid=0` flags
  frames where SMIRK or MediaPipe failed; long stretches of `valid=0`
  produce ugly interpolation.

## 1. Verify dataset + render full overlays

```sh
bash demo/01_preprocess_and_visualize.sh                            # train split, 60 frames
bash demo/01_preprocess_and_visualize.sh --split test --num-frames 30 --mp4
bash demo/01_preprocess_and_visualize.sh --pytorch3d
```

Verifies the dataset is loadable end-to-end and renders RGB + mask + DWpose +
FLAME-landmark overlays on top of the original image. Complementary to step 0's
`debug.mp4` (black background, smaller).

Output: `demo/output/01_overlay/<subject>/<split>/{NNNNN.png[, overlay.mp4]}`.

## 2. Train (personal adaptation)

```sh
bash demo/02_train_subject.sh                                       # subject 001, full
bash demo/02_train_subject.sh --quick                               # subset eval, fast turnaround
bash demo/02_train_subject.sh --epochs 5 --wandb-mode disabled
bash demo/02_train_subject.sh --pytorch3d
```

Output: `<repo>/../log/<subject>/configs/<conf-stem>/train/checkpoints/`.

## 3. Cross-reenactment

```sh
bash demo/03_cross_reenact.sh                                       # self-reenact 001 -> 001/test
bash demo/03_cross_reenact.sh --target Turnbull3                    # requires data/datasets/Turnbull3/
bash demo/03_cross_reenact.sh --quick --fast-test
bash demo/03_cross_reenact.sh --conf-reenact code/configs/reenact_002.conf
```

Output: `<repo>/../log/<source>/configs/<conf-stem>/train/eval[_quick]_reenact_<target>/test.mp4`.

## Notes

- `--pytorch3d` (steps 1 / 2 / 3) toggles the upstream PyTorch3D backend
  for that single invocation only (per `memo/pytorch3d_dual_mode.md`);
  default is the in-house path and works without `pytorch3d` installed.
- Step 0 dependencies (`mediapipe`, `controlnet_aux`, `mmcv`/`mmdet`/`mmpose`,
  RVM transitive deps) are pinned in `setup.sh`. Re-run `bash setup.sh
  --pip-only` after pulling this branch.
- Step 0 currently uses `bb_scale=2.0` which gives "face + shoulders"
  framing matching subject 001. Tighter face-only training datasets can
  override with `--bb-scale 1.4`.

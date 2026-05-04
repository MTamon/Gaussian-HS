# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reference implementation of *Gaussian Head & Shoulders* (arXiv 2405.12069) — anchor-Gaussian-guided neural upper-body avatars. The training/eval/reenactment code lives under `code/`; the project ships two CUDA submodules (`submodules/diff-gaussian-rasterization`, `submodules/simple-knn`) and depends on FLAME 2020/2023 assets that are downloaded separately.

This branch (`cuda128`) targets CUDA 12.8 / PyTorch 2.9.1 / Python 3.11 / RTX 5090 (sm_120 / Blackwell). It diverges from upstream in three structural ways: (1) a runtime-switchable in-house replacement for the small PyTorch3D surface this repo touches, (2) a `weights_only=False` wrapper around `torch.load`, and (3) deterministic pinned installs via `setup.sh`.

## Setup, data, and run commands

```sh
bash setup.sh                          # create conda env `gaussian-hs`, install pins, build submodules, sanity-check
bash setup.sh --pip-only               # install into the currently active env instead
bash setup.sh --no-assets              # skip download_assets.sh at the end

bash download_assets.sh --flame_user U --flame_pass P   # FLAME2020/2023 + subject 001 dataset
bash download_assets.sh --no_flame                      # dataset only
bash download_assets.sh --no_dataset                    # FLAME only

conda activate gaussian-hs

cd code
bash train.sh                          # train MLP + distilled, then full eval (writes under ../log/)
bash reenact.sh                        # cross-subject reenactment

# Fine-grained: dispatch through the entry point
python scripts/exp_runner.py --conf configs/ghs.conf --subject 001 --quick_eval
python scripts/exp_runner.py --conf configs/ghs.conf --subject 001 --is_eval --run_fast_test
python scripts/exp_runner.py --conf configs/ghs.conf --subject 001 --is_reenact --conf_reenact configs/reenact_002.conf
```

There is no test suite, linter, or formatter configured. The closest thing to a smoke check is the inline import block at the bottom of `setup.sh`.

`train.sh` writes experiments to `../log/<subject>/<methodname>/` (relative to `code/`, i.e. one level above the repo root). The dataset is expected at `../data/datasets/001/001/{train,test}/...`. `download_assets.sh` lays both out correctly.

## Architecture

`code/scripts/exp_runner.py` is the single CLI entry point. Based on flags it constructs one of three runners — `TrainRunner`, `TestRunner`, `ReenactRunner` — each owning its own data loading / wandb init / checkpoint I/O. There is no shared base class; flag dispatch and per-subject conf overrides happen inside `exp_runner.py` itself (e.g. subject 001 forces a separate test sub-directory; subject 003 patches a `distill_texture_bbox`).

The core model is `code/model/point_avatar_model.py::PointAvatar`, which composes:

- `flame.FLAME` — FLAME 2020 head model loaded from `code/flame/FLAME2020/{generic_model.pkl, landmark_embedding.npy}`.
- `model.deformer_network.ForwardDeformer` — LBS-based forward warp from canonical to posed space.
- `model.gaussian.gaussian_model.GaussianModel` — the 3D Gaussian point cloud (densify/prune/save_ply) from the standard 3DGS lineage.
- `model.gaussian.gaussian_renderer.render` — calls into `diff_gaussian_rasterization`.
- `model.layer.gs_img_model.GsImgNetwork` — the *anchor Gaussian + texture warping* head from the paper. Uses FPS for anchor init and Euler→matrix for anchor orientation; both go through `pytorch3d_compat`.

Loss assembly lives in `model.loss.Loss`; LR/loss-weight scheduling in `model.scheduler.{Constant,Linear,Exp,Sequential}Schedule`. VGG perceptual loss is a `model.vgg_feature` wrapper with a warm-up + linear ramp configured via `loss.vgg_*` keys.

Configs use HOCON (`pyhocon`). `code/configs/default.conf` is the base; `ghs.conf` extends it via `include required("./default.conf")`; `reenact_*.conf` are tiny overlay files merged on top via `ConfigTree.merge_configs`. Per-CLI-flag conf mutations (e.g. `--run_fast_test` → `affine_type=projective`, `test.opt_iter=50`) happen in `exp_runner.py` *after* the merge.

## Things specific to this branch

**PyTorch3D dispatcher (`code/model/pytorch3d_compat.py`).** The four PyTorch3D APIs the repo uses (`knn_points` K=1, `sample_farthest_points`, `euler_angles_to_matrix("XYZ")`, and the eval-time landmark rasterizer in `point_avatar_model.py`) have in-house pure-PyTorch equivalents that are numerically verified against upstream. Backend is selected at *import time* by the `GAUSSIAN_HS_USE_PYTORCH3D` env var — set to `1` to use upstream PyTorch3D, anything else (default) uses the in-house path. **Do not `export` this variable** in `~/.bashrc` or in `source`-d scripts; scope it to a single python invocation: `GAUSSIAN_HS_USE_PYTORCH3D=1 python scripts/test.py ...`. Once `pytorch3d_compat` has been imported the choice is frozen for that process. New renderer-style usages should follow the `if use_pytorch3d():` guard pattern in `point_avatar_model.py`. See `memo/pytorch3d_dual_mode.md`.

**Checkpoint loading.** PyTorch 2.6+ defaults `torch.load(weights_only=True)`, which rejects this repo's checkpoints (they contain optimizer/scheduler state and HOCON objects). Always load via `utils.torch_compat.load_trusted_checkpoint`, never raw `torch.load`. Only fall back to raw `torch.load(..., weights_only=True)` if the input is third-party / untrusted.

**`torch.meshgrid`.** Always pass `indexing='ij'` explicitly; the implicit default was removed.

**NumPy 2.x.** `numpy==2.2.6` is pinned. Don't reintroduce removed aliases (`np.float`, `np.int`, `np.bool`, `np.complex`) — use `np.float64` etc. directly. `chumpy==0.71` is built from `mattloper/chumpy@580566e` because the PyPI build is broken under NumPy 2; `setup.sh` enforces the version pin after install.

**CUDA build flags.** `setup.sh` exports `TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0;12.0` and `FORCE_CUDA=1`. The `12.0` is what enables sm_120 (RTX 5090). Use the same arch list when rebuilding `submodules/*` or installing pytorch3d from source.

**Pinned, no-deps installs.** Every Python package in `setup.sh` is installed with `--no-deps` against an exact pinned version; do not `pip install` anything new without pinning it the same way, or the resolver will pull conflicting transitive versions. `requirement.txt` is *not* the source of truth for this branch — `setup.sh` is. PyTorch / torchvision are pulled from the cu128 index (`https://download.pytorch.org/whl/cu128`).

## Notes for adding scripts

- Place new entry points under `code/scripts/` and prefer extending `exp_runner.py` flags over adding new top-level scripts. Runners are constructed with `**kwargs` from the parsed argparse namespace — adding a new option means threading it through `exp_runner.py` and the relevant runner's `__init__`.
- Use `partial(print, flush=True)` (existing pattern) if you need stdout to flush during long runs.
- Anything that loads a project checkpoint goes through `load_trusted_checkpoint`. Anything that touches PyTorch3D APIs goes through `pytorch3d_compat`. For demo scripts that need the upstream path, write `GAUSSIAN_HS_USE_PYTORCH3D=1 python ...` in the wrapper shell script, never `export`.

## Reference docs in repo

- `memo/pytorch3d_dual_mode.md` — backend dispatch design, env-var scoping rules, equivalence verification results.
- `memo/environment_notes.md` — pinned versions, sm_120 build notes, PyTorch3D source-build recipe if needed.
- `memo/code_change_log.md` — log of CUDA-12.8 / PyTorch-2.9 porting changes.
- `README.md` — public-facing setup + paper links.

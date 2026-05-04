# CLAUDE.md

## Project

Reference implementation of *Gaussian Head & Shoulders* (arXiv 2405.12069) — anchor-Gaussian-guided neural upper-body avatars. Training/eval/reenactment code lives under `code/`; two CUDA submodules under `submodules/`; FLAME 2020/2023 assets downloaded separately.

Branch `cuda128` targets CUDA 12.8 / PyTorch 2.9.1 / Python 3.11 / RTX 5090 (sm_120 / Blackwell). Key divergences from upstream: (1) runtime-switchable in-house PyTorch3D replacement (`pytorch3d_compat`), (2) `weights_only=False` wrapper via `load_trusted_checkpoint`, (3) deterministic pinned installs via `setup.sh`.

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

No test suite, linter, or formatter. Smoke check: inline import block at the bottom of `setup.sh`.

## Architecture

`code/scripts/exp_runner.py` is the single CLI entry point; constructs one of `TrainRunner` / `TestRunner` / `ReenactRunner`. No shared base class; per-subject conf overrides happen in `exp_runner.py` (e.g. subject 001 separate test dir, subject 003 patches `distill_texture_bbox`).

Core model `PointAvatar` (`code/model/point_avatar_model.py`) composes:

- `flame.FLAME` — FLAME 2020 head model
- `model.deformer_network.ForwardDeformer` — LBS forward warp
- `model.gaussian.gaussian_model.GaussianModel` — 3DGS point cloud
- `model.gaussian.gaussian_renderer.render` — calls `diff_gaussian_rasterization`
- `model.layer.gs_img_model.GsImgNetwork` — anchor Gaussian + texture warping; FPS init and Euler→matrix via `pytorch3d_compat`

Loss: `model.loss.Loss`. Scheduling: `model.scheduler.{Constant,Linear,Exp,Sequential}Schedule`. VGG perceptual loss via `model.vgg_feature` with warm-up + linear ramp (`loss.vgg_*` keys).

Configs: HOCON (`pyhocon`). `default.conf` is base; `ghs.conf` extends via `include required`; `reenact_*.conf` are overlays via `ConfigTree.merge_configs`. CLI flag mutations apply in `exp_runner.py` after merge.

## Branch-specific notes

**PyTorch3D dispatcher (`code/model/pytorch3d_compat.py`).** In-house pure-PyTorch equivalents for `knn_points` (K=1), `sample_farthest_points`, `euler_angles_to_matrix("XYZ")`, and the eval-time landmark rasterizer — numerically verified against upstream. Backend selected at import time via `GAUSSIAN_HS_USE_PYTORCH3D` env var (`1` = upstream, default = in-house). **Do not `export`** in `~/.bashrc` or sourced scripts — scope per invocation: `GAUSSIAN_HS_USE_PYTORCH3D=1 python scripts/test.py ...`. Choice is frozen after first import. New usages follow the `if use_pytorch3d():` pattern in `point_avatar_model.py`. See `memo/pytorch3d_dual_mode.md`.

**Checkpoint loading.** PyTorch 2.6+ defaults `weights_only=True`, rejecting this repo's checkpoints. Always use `utils.torch_compat.load_trusted_checkpoint`; never raw `torch.load` for project checkpoints.

**`torch.meshgrid`.** Always pass `indexing='ij'` explicitly.

**NumPy 2.x.** `numpy==2.2.6` pinned. Use `np.float64` etc. — removed aliases (`np.float`, `np.int`, `np.bool`, `np.complex`) must not be reintroduced. `chumpy==0.71` built from `mattloper/chumpy@580566e` (PyPI build broken under NumPy 2).

**CUDA build flags.** `TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0;12.0` and `FORCE_CUDA=1` (12.0 = sm_120 for RTX 5090). Use same arch list when rebuilding `submodules/*` or building pytorch3d from source.

**Pinned installs.** All packages installed `--no-deps` at exact pinned versions. Do not `pip install` without pinning. `setup.sh` is the source of truth (not `requirement.txt`). PyTorch/torchvision from `https://download.pytorch.org/whl/cu128`.

## Adding scripts

- New entry points under `code/scripts/`; prefer extending `exp_runner.py` flags over new top-level scripts.
- Use `partial(print, flush=True)` for flushed stdout in long runs.
- Project checkpoints → `load_trusted_checkpoint`. PyTorch3D APIs → `pytorch3d_compat`. Upstream path in demo scripts → `GAUSSIAN_HS_USE_PYTORCH3D=1 python ...` in the shell wrapper, never `export`.

## Reference docs

- `memo/pytorch3d_dual_mode.md` — backend dispatch design and equivalence verification
- `memo/environment_notes.md` — pinned versions, sm_120 build notes, PyTorch3D source-build recipe
- `memo/code_change_log.md` — CUDA-12.8 / PyTorch-2.9 porting log
- `README.md` — public-facing setup + paper links
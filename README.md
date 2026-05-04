# Gaussian Head & Shoulders: High Fidelity Neural Upper Body Avatars with Anchor Gaussian Guided Texture Warping

## [Paper](https://arxiv.org/abs/2405.12069) | [Project Page](https://gaussian-head-shoulders.netlify.app/) | [Data](https://drive.google.com/drive/folders/123DTRPc-Gfpl3pKbzmuNk4nBk72vWyS_?usp=sharing)

<img src="assets/teaser.png" width="650" height="223"/> 

## Getting Started
* Clone this repo recursively: `git clone git@ChikaYan/Gaussian-HS.git --recursive`
* Install the pinned CUDA 12.8 environment: `bash setup.sh`
* To install into an already active environment instead: `bash setup.sh --pip-only`
* To download FLAME assets and the subject 001 demo dataset: `bash download_assets.sh --flame_user USER --flame_pass PASS`
* Activate the environment before running: `conda activate gaussian-hs`

## Preparing dataset
Our data format is the same as [IMavatar](https://github.com/zhengyuf/IMavatar/). 

`download_assets.sh` downloads subject 3 from PointAvatar, renames it to `001`, and merges `001.zip` from the Gaussian-HS release, which contains the additional DWposes needed to run our method. The assembled dataset is placed under `../data/datasets/001/001`.



## PyTorch3D backend (in-house / upstream)

This branch ships an in-house drop-in replacement for the small PyTorch3D
surface this repository touches (`knn_points` with K=1, `sample_farthest_points`,
`euler_angles_to_matrix("XYZ")`, and the evaluation-time landmark rasterizer).
It is selected by default and works on CUDA 12.8 / RTX 5090 (sm_120) without
needing PyTorch3D itself to be installed.

To switch to the upstream PyTorch3D implementation at runtime, set the
environment variable when launching:

```sh
GAUSSIAN_HS_USE_PYTORCH3D=1 python scripts/test.py ...
```

Keep this variable scoped to the invoking shell script (do not `export` it
in `~/.bashrc` or via `source`-ed scripts) so it does not leak. See
[memo/pytorch3d_dual_mode.md](memo/pytorch3d_dual_mode.md) for the full
rationale and demo-script guidelines, and
[memo/environment_notes.md](memo/environment_notes.md) for the
`TORCH_CUDA_ARCH_LIST=12.0` build setting required for sm_120.

## Training and Testing

To train and evaluate both the MLP and distilled version:
```
cd code; bash train.sh
```

To run reenactment:

```
cd code; bash reenact.sh
```

## TODO List

- [x] Release training code
- [x] Release reenactment code
- [ ] Release data preprocess code

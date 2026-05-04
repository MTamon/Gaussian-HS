# demo/

Minimal end-to-end demo scripts for Gaussian-HS. See top-level `README.md` and
`CLAUDE.md` for context (full docs to be filled in separately).

Each script supports `--help` for the full flag list.

## 1. Preprocess + visualize features

```sh
bash demo/01_preprocess_and_visualize.sh                            # train split, 60 frames
bash demo/01_preprocess_and_visualize.sh --split test --num-frames 30 --mp4
bash demo/01_preprocess_and_visualize.sh --pytorch3d
```

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

- Activate the env first: `conda activate gaussian-hs`.
- `--pytorch3d` toggles the upstream PyTorch3D backend for that single
  invocation only (per `memo/pytorch3d_dual_mode.md`); default is the in-house
  path and works without `pytorch3d` installed.
- Feature extraction from raw video (DWpose detection / FLAME tracking) is in
  the upstream TODO list and not in this repo. Step 1 verifies the prepared
  dataset rather than producing it; run `bash download_assets.sh` first.

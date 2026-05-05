# demo/

[English](README.md)

Gaussian-HS のエンドツーエンドデモスクリプトです。プロジェクト全体の
背景は、トップレベルの `README.md` と `CLAUDE.md` を参照してください。

このパイプラインは4ステップです。生動画から被験者ごとのデータセットを
作成し、アラインメントを確認し、個人モデルを学習して、別の被験者から
再演します。

```text
mp4  ──▶  00_preprocess_video.sh  ──▶  data/datasets/<subj>/...
                                          │
                                          ├──▶  01_preprocess_and_visualize.sh   (sanity check)
                                          ├──▶  02_train_subject.sh              (training)
                                          └──▶  03_cross_reenact.sh              (re-enactment)
```

各スクリプトは、完全なフラグ一覧を表示する `--help` に対応しています。
先に環境を有効化してください: `conda activate gaussian-hs`。

## 0. 生動画からデータセットを作成する (SMIRK + RVM + DWpose)

```sh
bash demo/00_preprocess_video.sh --video clip.mp4 --subject 999            # train split, 25 fps, 512px
bash demo/00_preprocess_video.sh --video clip.mp4 --subject 999 --split test
bash demo/00_preprocess_video.sh --video clip.mp4 --subject 999 --start-stage 7 --end-stage 9
```

`code/scripts/preprocess_smirk.py` のラッパーです。ステップ1-3でそのまま
使える IMavatar 形式のディレクトリを出力します。

```text
data/datasets/<subject>/<subject>/<split>/
    image/00000.png ...        # 正方形クロップ (S × S)
    mask/00000.png  ...        # RVM alpha matte
    semantic/00000.png ...     # face-parsing (CelebAMask-HQ labels)
    dwpose/00000.npy ...       # DWpose keypoints (IMavatar dict format)
    flame_params.json          # shape100 + per-frame expression50/pose15/world_mat
    debug.mp4                  # 黒背景の sanity overlay (下記参照)
    _work/                     # 中間生成物 (raw frames, bbox, smirk_flame.pt, eyes_pose.npz)
```

9個のステージは `--start-stage N --end-stage M` で個別に再実行できます。
単一コンポーネントを反復調整するときに便利です。

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

### 外部アセット

このスクリプトは、既存のサードパーティスタックを呼び出します。下記の
デフォルト位置に配置または clone してください。対応するフラグでパスを
上書きすることもできます。

- `--smirk-repo` `/home/mikawa/lab/outcome/smirk` -
  [`MTamon/smirk@release/cuda128`](https://github.com/MTamon/smirk/tree/release/cuda128)。
  `pretrained_models/SMIRK_em1.pt` に重みが必要です。
- `--hr-preprocess` `/home/mikawa/lab/outcome/HRAvatar/preprocess` -
  HRAvatar の `submodules/RobustVideoMatting/` (`rvm_resnet50.pth`) と
  `submodules/face-parsing.PyTorch/res/cp/` (`79999_iter.pth`) が必要です。
  初回に HRAvatar 内で `bash download_assets.sh` を実行して取得してください。
- DWpose の重み (yolox + dw-ll_ucoco_384) は、初回実行時に
  `controlnet_aux` 経由で `~/.cache/torch/hub/checkpoints/` に自動ダウンロード
  されます。

### FLAME パラメータの作られ方

| Parameter      | Source                                                           |
|----------------|------------------------------------------------------------------|
| `shape_params` (100) | SMIRK shape。有効フレームの平均を取り、先頭100次元にスライス |
| `expression`   (50)  | SMIRK exp。そのまま使い、無効フレームは線形補間 |
| `pose[0:3]`    (global) | SMIRK pose                                            |
| `pose[3:6]`    (neck)   | zeros (SMIRK は首を推定しません)                  |
| `pose[6:9]`    (jaw)    | SMIRK pose                                            |
| `pose[9:15]`   (L/R eye)| MediaPipe ARKit blendshapes (`eyeLook*`/`eyeBlink*`) → rot6d → axis-angle。SMIRK とは独立に計算するため、SMIRK 内部の `--with_eye_pose` 再クロップがカメラ座標を乱しません。ゼロにフォールバックするには `--no-eye-pose` を渡してください。 |
| `world_mat`    (4×4)    | SMIRK の ortho cam (`s, tx, ty`) から作る DECA 風 pseudo-perspective。`focal_pix = 5000 · S / 224`。回転は identity、並進はフレームごと。 |
| `intrinsics`   (4)      | S に関係なく正規化 pinhole `[5000/224, 5000/224, 0.5, 0.5]` |

`world_mat` と `flame_pose` は、学習時に `optimize_camera=True` /
`optimize_pose=True` でフレームごとに refine されます。そのため初期値は
おおよそ正しければ十分です。

### debug 動画の sanity check

`debug.mp4` は、静かに混入するバグを見つける最も効率的な確認方法です。
新しい被験者ごとに一度確認してください。

- **赤い点 (FLAME 68 landmarks)** - 顔上に乗っているべきです。顔から外れる
  場合は、`world_mat` と画像のアラインメントが間違っています。
- **シアンの円 (DWpose nose / neck / shoulders)** - 対応する身体部位に
  乗っているべきです。身体から外れる場合は、DWpose 検出に失敗しているか、
  bbox crop で肩が隠れています。
- **ステータステキスト (frame index, smirk_valid, cam tuple)** - `valid=0`
  は SMIRK または MediaPipe が失敗したフレームです。`valid=0` が長く続くと
  補間結果が崩れます。

## 1. データセット検証と全 overlay のレンダリング

```sh
bash demo/01_preprocess_and_visualize.sh                            # train split, 60 frames
bash demo/01_preprocess_and_visualize.sh --split test --num-frames 30 --mp4
bash demo/01_preprocess_and_visualize.sh --pytorch3d
```

データセットがエンドツーエンドで読み込めることを検証し、元画像の上に
RGB + mask + DWpose + FLAME landmark overlay をレンダリングします。
ステップ0の `debug.mp4` (黒背景で小さめ) を補完する確認です。

出力: `demo/output/01_overlay/<subject>/<split>/{NNNNN.png[, overlay.mp4]}`。

## 2. 学習 (personal adaptation)

```sh
bash demo/02_train_subject.sh                                       # subject 001, full
bash demo/02_train_subject.sh --quick                               # subset eval, fast turnaround
bash demo/02_train_subject.sh --epochs 5 --wandb-mode disabled
bash demo/02_train_subject.sh --pytorch3d
```

出力: `<repo>/../log/<subject>/configs/<conf-stem>/train/checkpoints/`。

## 3. Cross-reenactment

```sh
bash demo/03_cross_reenact.sh                                       # self-reenact 001 -> 001/test
bash demo/03_cross_reenact.sh --target Turnbull3                    # requires data/datasets/Turnbull3/
bash demo/03_cross_reenact.sh --quick --fast-test
bash demo/03_cross_reenact.sh --conf-reenact code/configs/reenact_002.conf
```

出力: `<repo>/../log/<source>/configs/<conf-stem>/train/eval[_quick]_reenact_<target>/test.mp4`。

## Notes

- `--pytorch3d` (ステップ1 / 2 / 3) は、その1回の実行だけ upstream の
  PyTorch3D backend に切り替えます (`memo/pytorch3d_dual_mode.md` 参照)。
  デフォルトは in-house path で、`pytorch3d` がインストールされていなくても
  動作します。
- ステップ0の依存関係 (`mediapipe`, `controlnet_aux`, `mmcv`/`mmdet`/`mmpose`,
  RVM の推移的依存関係) は `setup.sh` で pin されています。このブランチを
  pull した後に `bash setup.sh --pip-only` を再実行してください。
- ステップ0は現在 `bb_scale=2.0` を使います。これは subject 001 に合わせた
  「顔 + 肩」のフレーミングです。顔だけに近いタイトな学習データセットでは
  `--bb-scale 1.4` で上書きできます。

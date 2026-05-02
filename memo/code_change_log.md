# 主な改修内容と検証結果

CUDA 12.8 / PyTorch 2.9 / Python 3.11 / RTX 5090 (sm_120) 対応に伴う改修の
要点を記録する。詳細な切替方針は `pytorch3d_dual_mode.md`、環境前提は
`environment_notes.md` を参照。

## 1. pytorch3d 二系統運用 (内製 / 上流) への変更

* `code/model/pytorch3d_compat.py` を環境変数 `GAUSSIAN_HS_USE_PYTORCH3D`
  によるディスパッチャに統合。デフォルト（未設定）は内製、`1` で上流。
* `code/model/point_avatar_model.py` の評価時ランドマーク描画も同フラグで
  分岐。pytorch3d 経路では `PerspectiveCameras + PointsRasterizer +
  AlphaCompositor` を使い、内製経路では intrinsics による直接投影 + 3x3
  ピクセルブロック描画を使う。
* `code/model/layer/gs_img_model.py` 等の anchor 初期化と Euler 行列生成は、
  `pytorch3d_compat` 経由のディスパッチで自動的に二系統対応。

### 内製実装の同等性検証

PyTorch3D 上流の Python 参照実装を移植して比較した結果（float64, 5 trial 以上）:

| 関数 | 検証結果 |
| --- | --- |
| `knn_points` (K=1, K=4, P1=4097 のチャンク境界含む) | index 完全一致、距離は 1e-15 オーダーの丸め差のみ |
| `sample_farthest_points` (P=2000, K=256 / P=50000, K=512) | 選択 index 列が `sample_farthest_points_naive` と完全一致 |
| `euler_angles_to_matrix` ("XYZ", "ZYX", "YXZ", "XZY") | 上流とビット一致 (max diff 0.0) |

K > P のときの padding（内製: 最後の index を埋める / 上流: -1 を埋める）
だけ API 差分があるが、本リポ内の使用ケース (`N_anchor < num_points`) では
発生しない。

ランドマーク描画は loss に寄与しない評価時可視化のため、pytorch3d 経路と
内製経路で見た目の精度は完全一致しない（disk vs 3x3 ブロック）が、デバッグ
可視化用途には十分。

## 2. `torch.load` の `weights_only` 対応

PyTorch 2.6 以降は `torch.load` の `weights_only` デフォルトが `True` に
変更されたため、optimizer state や HOCON 等の Python オブジェクトを含む
チェックポイントが読めなくなる。

* `code/utils/torch_compat.py::load_trusted_checkpoint` を導入し、
  `weights_only=False` を明示する wrapper として運用。
* `code/scripts/{train,test,distill,reenact}.py` の `torch.load` をすべて
  この wrapper 経由に置き換え済み。
* 新規スクリプトを追加するときも、信頼できる自前のチェックポイントを
  読むなら `load_trusted_checkpoint` を使うこと。第三者由来の pickle を
  読む用途では `torch.load(..., weights_only=True)` のままにする。

参考: <https://docs.pytorch.org/docs/stable/generated/torch.load.html>

## 3. `torch.meshgrid` の `indexing` 明示

`torch.meshgrid` は `indexing` 引数が必須化された。本リポ内の該当箇所
(`code/datasets/real_dataset.py` 等) を `indexing='ij'` 明示済み。

## 4. NumPy 2 系対応

`numpy==2.2.6` を前提として:

* `np.float`, `np.int`, `np.bool`, `np.complex` 等の旧 alias は本コード内に
  使用なし（確認済み）。新規コードでも使わない。
* `chumpy` は PyPI 版 (0.69 など) では NumPy 2 系で動かないため、ソース
  ビルド版 0.71 を使う前提。

## 5. 依存ライブラリ

`requirement.txt` を以下に固定:

```
imageio==2.34.2
lpips==0.1.4
matplotlib==3.9.2
opencv-python==4.10.0.84
Pillow==10.4.0
pyhocon==0.3.59
scikit-image==0.25.2
scipy==1.15.3
trimesh==4.4.9
chumpy==0.71
wandb==0.17.8
protobuf==4.25.5
numpy==2.2.6
pandas==2.2.3
plyfile==1.1
av==12.3.0
einops==0.8.0
```

PyTorch / pytorch3d / CUDA サブモジュール (`diff-gaussian-rasterization`,
`simple-knn`) は requirement.txt に含めず、`setup.sh` で個別ビルドする想定。

## 残タスク

* `setup.sh` の作成（PyTorch 2.9+CUDA 12.8 のインストール、CUDA 拡張の
  sm_120 ビルド、pytorch3d を入れる場合の `TORCH_CUDA_ARCH_LIST=12.0`
  指定、chumpy ソースビルド）。
* `submodules/diff-gaussian-rasterization` と `submodules/simple-knn` の
  sm_120 ビルド確認（実機ビルドが必要、未着手）。
* デモスクリプト類は `pytorch3d_dual_mode.md` の env var スコーピング指針に
  沿って作成すること。

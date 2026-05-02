# pytorch3d 内製 / pytorch3d 二系統運用

## 背景

本リポジトリでは元々 `pytorch3d` の以下 4 種類の API を使用していた:

| API | 使用箇所 | 用途 |
| --- | --- | --- |
| `pytorch3d.ops.knn_points` | `code/model/point_avatar_model.py` | FLAME ロスでの最近傍探索 (K=1) |
| `pytorch3d.ops.sample_farthest_points` | `code/model/layer/gs_img_model.py` 他 | anchor 初期化 (FPS) |
| `pytorch3d.transforms.euler_angles_to_matrix` | `code/model/layer/gs_img_model.py` 他 | "XYZ" Euler → 回転行列 |
| `pytorch3d.renderer` (PerspectiveCameras / PointsRasterizer / AlphaCompositor) + `pytorch3d.structures.Pointclouds` | `code/model/point_avatar_model.py` | 評価時のランドマーク可視化（loss には不関与） |

CUDA 12.8 / PyTorch 2.9 / sm_120 (RTX 5090, Blackwell) 環境では、執筆時点で
pytorch3d の公式 wheel は提供されておらず、上流リポジトリにも:

* #1949 PyTorch 2.6 対応未解決 (Open)
* #1962 RTX 5090 + CUDA 12.8 + PyTorch 2.8 ビルド (Closed だが解決手順未記載)
* #1966 RTX 50x + CUDA 12.8 への成功事例なし (Open)
* #1970 PyTorch 2.7 + CUDA 12.8 で NVCC ヘッダエラー (Open)
* #2011 CUDA 13.0 でリンクエラー (Open)

といった未解決の互換性課題が残る。一方、本リポが利用している API 自体は枯れた
コアの Python+CUDA 関数群で、**ビルドさえ通れば動作上の問題はない**ことを
確認済み（`memo/code_change_log.md` の検証結果を参照）。

## 採用方針

**実行時に環境変数で切り替えられるようにし、両方の経路を 1 ソースに保持する。**

* デフォルト（環境変数未設定 / `0`）: 内製実装を使用
* `GAUSSIAN_HS_USE_PYTORCH3D=1` を設定: 上流 pytorch3d を使用

切替の中心は `code/model/pytorch3d_compat.py` のディスパッチャ。
`use_pytorch3d()` を呼ぶことで現在のモードを問い合わせられる。
ランドマーク描画 (`code/model/point_avatar_model.py`) もこのフラグで
PyTorch3D の `PointsRasterizer + AlphaCompositor` パスと内製簡易投影パスを
切り替える。

### 内製パスと pytorch3d パスの等価性

`code/model/pytorch3d_compat.py` 内の内製実装は、本リポジトリでの利用形態
（`knn_points` K=1 / `sample_farthest_points` 既定開始点 / `XYZ` のみ）に
ついて、PyTorch3D 上流 Python 参照実装と数値的に等価であることを
事前に確認している。

* `knn_points` (K=1, K=4): index 完全一致、距離 1e-15 オーダーの float64 丸め差のみ
* `sample_farthest_points`: 開始点 0 / `argmax` tie-break が上流と一致し、選択 index 列も完全一致
* `euler_angles_to_matrix("XYZ")`: 上流とビット一致 (max diff 0.0)

ランドマーク描画は loss に関わらない評価時可視化のため、内製版は
`PointsRasterizer` の disk ラスタライズの代わりに 3x3 ピクセルブロック
描画で代用している。完全な見た目互換ではないが、デバッグ用途には十分。

## デモスクリプト作成時の指針

### 環境変数の流出を防ぐ

`GAUSSIAN_HS_USE_PYTORCH3D` は **シェルスクリプト内に閉じ込めて呼び出すこと**。
親シェルへ漏らさないため、スクリプトの先頭で `export` するのではなく、
`env VAR=val` 形式または `VAR=val cmd` 形式を使う:

```sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/code"

# pytorch3d 経路を使う場合（呼び出した python プロセスの寿命でのみ有効）
GAUSSIAN_HS_USE_PYTORCH3D=1 python scripts/test.py "$@"
```

`source` 不要のスクリプトとして実行 (`bash demo.sh` または `./demo.sh`) する
こと。`source demo.sh` で実行すると `export` した変数が呼び出し元シェルに
残るため避ける。

### 既存 `train.sh` / `reenact.sh` の扱い

既存の `code/train.sh`, `code/reenact.sh` は環境変数を export しない素の
`python` 起動なので、デフォルト（内製）で動く。pytorch3d 経路で動かしたい
場合は、対応スクリプトのコピーを作って先頭の python 起動行だけ
`GAUSSIAN_HS_USE_PYTORCH3D=1 python ...` の形に書き換える。

### 切替の確認

切替が効いているかを起動時にログ出力したい場合、`code/scripts/*.py` の
冒頭で次のようにモードを表示できる:

```python
from model.pytorch3d_compat import use_pytorch3d
print(f"[backend] pytorch3d={use_pytorch3d()}")
```

## ディスパッチャ追加 API メモ

`code/model/pytorch3d_compat.py` で公開している関数:

* `use_pytorch3d() -> bool` — 現在のバックエンドフラグ
* `knn_points(p1, p2, K=1, return_nn=False)` — 戻り値は `(dists, idx, nn|None)` のタプル
* `sample_farthest_points(points, K)` — 戻り値は `(sampled, indices)` のタプル
* `euler_angles_to_matrix(euler_angles, convention)` — 内製パスは "XYZ" 等の重複なし 3 文字のみ受理

`pytorch3d.renderer.*` は API 呼び出し側 (`point_avatar_model.py`) で
`use_pytorch3d()` ガード付きの import としてある。レンダラ系を新たに
使う場合は同じパターンを踏襲すること。

## 切替の今後の拡張

* CLI 引数で切り替えたい場合: `argparse` のフラグで env var を上書き設定
  し、`pytorch3d_compat` を import する **前** に `os.environ` を設定する
  必要がある。`code/scripts/*.py` の最上段で実装するのが安全。
* 一度 `pytorch3d_compat` を import したあと env var を変えても切替は
  反映されない（モジュール ロード時に確定する）。

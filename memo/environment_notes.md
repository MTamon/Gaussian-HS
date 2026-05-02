# 目標環境と依存メモ

## ターゲット環境

| 項目 | バージョン |
| --- | --- |
| OS | Linux |
| GPU | RTX 5090 (Blackwell, 計算能力 sm_120) |
| CUDA Toolkit | 12.8 |
| C コンパイラ | gcc 11 (chumpy / 各 CUDA 拡張のビルドで使用) |
| Python | 3.11 |
| PyTorch | 2.9 (CUDA 12.8 ビルド) |
| NumPy | 2.2.6 |
| chumpy | 0.71 (ソースビルド) |

すべて固定バージョン運用。`pip install --no-deps` で個別固定する想定。
依存解決はしない。

## requirement.txt の扱い

* `requirement.txt` には `--no-deps` でインストールされる Python パッケージ
  本体だけを列挙する。
* PyTorch / pytorch3d / 各 CUDA 拡張サブモジュール (`diff-gaussian-rasterization`,
  `simple-knn`) は requirement.txt には含めず、`setup.sh`（後で作成）の中で
  個別に `pip install` する。
* `numpy==2.2.6` を陽に固定。NumPy 2 系で消滅した `np.float` / `np.int` 等の
  alias は本コード内では使われていないことを確認済み。
* `chumpy==0.71` は NumPy 2 系では公式 PyPI 版が動かないため、
  `pip install --no-build-isolation --no-deps git+https://...` などで
  ソースビルドする。gcc 11 が要る。

## pytorch3d インストール

`memo/pytorch3d_dual_mode.md` の通り、デフォルトの内製パスでは pytorch3d は
不要。`GAUSSIAN_HS_USE_PYTORCH3D=1` で切り替えた場合のみ pytorch3d が必要。

CUDA 12.8 / PyTorch 2.9 / sm_120 では公式 wheel は提供されていないので、
ソースビルドする。`setup.sh` 作成時に以下を含める想定:

```sh
# pytorch3d をソースビルド (任意。内製パスのみで運用するなら不要)
export TORCH_CUDA_ARCH_LIST="12.0"   # sm_120 を有効化
pip install --no-deps --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
```

`TORCH_CUDA_ARCH_LIST` には RTX 5090 用に `"12.0"` を含める必要がある。
ビルド失敗時は GitHub issue #1970 のヘッダ起因エラーを踏む可能性があるので、
PyTorch3D の main ブランチ最新を試す or 該当パッチを当てる必要がある。

## CUDA サブモジュール

`submodules/diff-gaussian-rasterization` と `submodules/simple-knn` は
sm_120 用にビルドする必要がある。`setup.sh` で:

```sh
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-deps submodules/diff-gaussian-rasterization
pip install --no-deps submodules/simple-knn
```

これらのサブモジュールに sm_120 対応のソース修正が必要かどうかは、本作業
ブランチ時点では未確認（リポ内に submodule の中身が無いため）。setup.sh
作成時に実機で初回ビルドを試して、必要に応じてフォーク版に差し替えること。

## Python バージョン依存の API 変更

* `torch.load` は PyTorch 2.6 以降 `weights_only=True` がデフォルト。
  本リポのチェックポイントは optimizer state など Python オブジェクトを
  含むため、`code/utils/torch_compat.py::load_trusted_checkpoint` 経由で
  `weights_only=False` を明示している。新規スクリプトを書く場合も同 wrapper を使う。
* `torch.meshgrid` は `indexing` 引数が必須化されている。本コードでは
  `indexing='ij'` を明示済み。新規コードでも同様に明示すること。
* `numpy` 2 系では `np.float`, `np.int`, `np.bool` 等の alias が削除済み。
  本コード内では未使用。新規コードでも `np.float64` 等を直接使うこと。

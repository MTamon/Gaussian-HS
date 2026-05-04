# Training クラッシュ: 全 Gaussian prune → reshape(0, -1) クラッシュ

CUDA 12.8 / PyTorch 2.9 / RTX 5090 (sm_120) 環境で `bash demo/02_train_subject.sh --quick` を
subject 001 で実行すると、訓練 600 iter 程度で次のエラーが発生する。デモスクリプト本体ではなく
upstream の挙動差由来の問題なので、別タスクで調査する想定でメモを残す。

## 症状

```
configs/ghs [49]  (50/70000):  loss: 0.267  rgb_loss: 0.354  ...
configs/ghs [99]  (100/70000): loss: 0.0274 rgb_loss: 0.0602 ...
   ...
configs/ghs [549] (550/70000): loss: 0.00609 ...
new points: 0
configs/ghs [599] (600/70000): loss: 0.00549 ...
Traceback (most recent call last):
  File "code/scripts/exp_runner.py", line 102, in <module>
    runner.run()
  File "code/scripts/train.py", line 622, in run
    model_outputs = self.model(...)
  File "code/model/point_avatar_model.py", line 263, in forward
    pose_feature=pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1),
RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1] because the
unspecified dimension size -1 can be any value and is ambiguous
```

* `loss` は 0.267 → 0.005 と順調に低下しており、dataloader / 前処理 / forward / backward は
  きちんと動いている。
* クラッシュ直前のログに `new points: 0` が出ている。densify ステップで増えた点が 0、
  かつ次の forward で `n_points = self.pc.points.shape[0] == 0` が成立 = **すべての Gaussian が
  prune された** とほぼ確定。
* この時点で anchor (`gs_img_model`) はまだ初期化されていない（`anchor_init_iter = 4000`）。

## 原因の二層構造

1. **密度化/prune の挙動差（根本原因）**
   * `code/model/gaussian/gaussian_model.py` の `densify_and_prune` は不透明度・スケール基準で
     Gaussian を削除する。CUDA 12.8 / PyTorch 2.9 への移植で、float の演算順序や
     `diff-gaussian-rasterization` (sm_120 ビルド) の勾配出力に微小な差が出ており、
     旧環境では残っていた Gaussian が一括 prune されてしまう、と推測。
   * 該当コンフィグ: `code/configs/ghs.conf` の
     ```hocon
     gs_opt {
         densify_grad_threshold = 0.0008
         densify_grad_threshold_vgg_warmup = 0.00025
         densify_until_iter = 25000
     }
     ```
   * 600 iter 時点では VGG warmup (`vgg_loss_warm_up = 10000`) より前のため、
     `densify_grad_threshold_vgg_warmup = 0.00025` が効いている。

2. **PyTorch 2.x の strict reshape (顕在化箇所)**
   * `tensor.reshape(0, -1)` は旧 PyTorch では `(0, 0)` を返したが、PyTorch 2.x で
     `cannot reshape tensor of 0 elements ... is ambiguous` を投げるようになった。
   * `code/model/point_avatar_model.py:263-266` の 4 つの reshape (`pose_feature`, `betas`,
     `transformations`, `transform_rot`) で同じパターン。さらに後段の
     `transformed_points.reshape(batch_size, n_points, 3)` 等でも同様。

## 調査の出発点

### A. 密度化を緩めて再現確認

最小修正で「prune が暴走しない」設定で動くかを確認する。

* `code/configs/ghs.conf` を一時的に書き換え:
  ```hocon
  gs_opt {
      densify_grad_threshold = 0.0001
      densify_grad_threshold_vgg_warmup = 0.0001
      densify_until_iter = 25000
  }
  ```
* または `code/model/gaussian/gaussian_model.py` の `prune_points` を一時的にバイパスして
  確かに「prune の暴走」が原因かを切り分ける。

### B. クラッシュ箇所のサバイバル化（応急処置）

`point_avatar_model.py:263-266` の 4 ヶ所 (および `:268-269` の reshape 2 ヶ所)
を `.flatten(0, 1)` ベースに置き換えると、PyTorch 2.x でも 0 要素テンソルを通せる:

```python
# 変更前
pose_feature.unsqueeze(1).expand(-1, n_points, -1).reshape(total_points, -1)

# 変更後 (n_points=0 でも安全)
pose_feature.unsqueeze(1).expand(-1, n_points, -1).flatten(0, 1)
```

ただし**これは A の根本原因を治さず、空の Gaussian クラウドで forward を続けるだけ**。
出力画像が真っ白になり学習にならない。あくまで「クラッシュを止めて anchor 初期化
(`iter = 4000`) まで到達できるか」を確認するためのデバッグ手段。

### C. diff-gaussian-rasterization の出力比較

`submodules/diff-gaussian-rasterization` を sm_120 でビルドし直したことで、勾配計算の
精度が変わっている可能性。次を比較する:

* `gaussians.viewspace_point_grad` の分布（旧環境ログがあれば対応 iter で比較）。
* `densify_and_prune` 内で削除された点数の log を `print` 等で吐き出し、どの基準
  （opacity 閾値 / scale 閾値 / view-space radius）で消えているかを特定。

### D. `radius_factor` / 初期点数

`ghs.conf` の `point_cloud { n_init_points = 10000, radius_factor = 0.75 }` は
初期 Gaussian を半径 0.75 の球内に一様分布させる。FLAME 顔の正規化スケールと
合わない場合、初期段階から大半が「カメラから見えない」位置にあり、 radii 0 で
prune される可能性。`radius_factor` を 0.5 / 0.3 等に下げて挙動が変わるか確認する。

## 関連: 既に治した周辺バグ

このクラッシュとは別件で、デモ起動時に踏んだ既知の API 互換問題は、デモ作業の中で
直接修正済み。再発に備えて記録:

| 症状 | 原因 | 修正 |
| --- | --- | --- |
| `wandb.init(config=ConfigTree(...))` で `ConfigMissingException: 'No configuration setting found for key _type'` | wandb 0.17+ が config の各値で `v.get("_type")` を default なしに呼ぶが、pyhocon `ConfigTree.get` は dict 非互換 | `code/utils/wandb_compat.py::to_plain_dict` を追加し、`code/scripts/{train.py,test.py}` の `wandb.init(config=...)` を `to_plain_dict(self.conf)` 経由に変更 |
| `imageio.imread(path, as_gray=True)` で `TypeError: The keyword as_gray is no longer supported.` | imageio 2.37 で `as_gray` が削除（後継は `mode='L'` / `mode='F'`） | `code/datasets/real_dataset.py:476,486`, `code/scripts/metric_mask_pred.py:120,129`, `code/utils/metrics.py:270,279` を `mode='L'` に統一 |
| `default.conf` の `data_folder = ../data/datasets`, `exps_folder = ../log/` が `cd code/` cwd 想定で repo 内を指してしまい、`download_assets.sh` が置く `<repo>/../data/datasets/` と整合しない | upstream の cwd アンカー違い | デモ側で `--data-root` / `--log-dir` 引数 + temp conf で `dataset.data_folder` / `train.exps_folder` を絶対パスに override（`demo/02_train_subject.sh`, `demo/03_cross_reenact.sh`）。本体コードはそのまま |

これらは demo の実行で「training の forward に到達するまで」の障害だったので、本件
(densifier 全 prune) はそれらを潰した先の本丸。

## 解決: 真の根本原因は densifier ではなく `load_mask` の二重正規化

調査の過程で、上記「原因の二層構造」のうち **(1) 密度化/prune の挙動差** は
誤った推測だったことが判明。3DGS upstream の `densify_and_prune` 自体は
壊れておらず、白一色のデータセットを与えられた結果として全 Gaussian の
opacity が 0 に収束し、当然のように prune されていただけだった。

### 真の原因 (確定)

`code/datasets/real_dataset.py` の `load_mask` (`as_gray=True` → `mode='L'` 移行で
発生したサイレントバグ):

```python
def load_mask(path, img_res):
    alpha = imageio.imread(path, mode='L')   # uint8 [0, 255]
    alpha = skimage.img_as_float32(alpha)    # uint8 → /255 → float32 [0, 1]
    alpha = cv2.resize(alpha, ...)            # still [0, 1]
    object_mask = alpha / 255                 # ★二重正規化 → [0, 0.00392]
    return object_mask
```

旧 `as_gray=True` は **float64 [0, 255]** を返し、`skimage.img_as_float32` は
**float 入力に対しては rescale しない** (cast のみ) ため末尾の `/ 255` で
ちょうど [0, 1] に収まっていた。`mode='L'` への移行で imageio が **uint8** を
返すようになり `img_as_float32` が新たに `/ 255` で正規化するように動作変更
されたため、末尾の `/ 255` が二重に作用して mask が事実上ゼロになった。

実測 (subject 001 のマスクで):

```
imread mode=L:           uint8   min=0  max=255  mean=113.04
after img_as_float32:    float32 min=0  max=1.0  mean=0.443
after final /255 (bug):  float32                 max=0.00392
```

### クラッシュとの因果

`real_dataset.py:267` の alpha-blend
```python
rgb = rgb * object_mask + (1 - object_mask)
```
で `object_mask ≈ 0` だと `rgb ≈ 1` → 訓練データ全画像が白一色 → 純白を出すだけの
自明解で loss は下がる (0.267 → 0.005) が、不透明な Gaussian は不要なので
opacity が全点で 0 方向に最適化される → iter=500 の `reset_opacity()` で
≤ 0.01 にクランプ → iter=600 の `densify_and_prune(0.005)` で全滅 →
次 forward で `n_points=0` の `reshape(0, -1)` クラッシュ。

### 修正

```diff
 def load_mask(path, img_res):
     alpha = imageio.imread(path, mode='L')
     alpha = skimage.img_as_float32(alpha)
     alpha = cv2.resize(alpha, (int(img_res[0]), int(img_res[1])))
-    object_mask = alpha / 255
+    object_mask = alpha
     return object_mask
```

合わせて、PyTorch 2.x の strict reshape に対する保険として
`code/model/point_avatar_model.py:263-264` の `.reshape(total_points, -1)` を
`.flatten(0, 1)` に置換 (n_points=0 でも well-defined にする防御策。本来は
今回の修正で n_points=0 にはならない)。

### 横展開チェック (修正不要)

| ファイル | 状況 |
| --- | --- |
| `code/scripts/metric_mask_pred.py:120,129` | `mode='L'` 後 `cv2.resize` のみ。`img_as_float32` も末尾 `/255` も無く戻り値 uint8 を整数 class ID として使用。挙動不変、修正不要 |
| `code/utils/metrics.py:270,279` | 同上 |
| `code/datasets/real_dataset.py:466-472` (`load_rgb`) | uint8 RGB → `img_as_float32` で [0, 1]、末尾 `/255` 無し。正しい |

つまり二重正規化は `load_mask` 1 箇所だけのローカルバグだった。

### 教訓

- `as_gray=True` → `mode='L'` の機械的置換は **戻り dtype が float → uint8 に
  変わる** という副作用があり、後段の `skimage.img_as_float32` の挙動が
  dtype 依存で変わる (float 入力は pass-through、uint 入力は /255)。
- 既に末尾正規化を持つ経路 (`load_mask` の `alpha / 255`) と組み合わさると
  二重正規化を起こす。
- 同種の機械的マイグレーションを行う際は **戻り値の値域を実データで検証** する
  (今回は `imread → img_as_float32 → /255` の各段で min/max を見ることで局所化できた)。
- loss が下がっても出力が collapse 解 (純白) に張り付いているだけ、という
  ケースがあるので、「loss が下がっている = 学習が進んでいる」と早合点しない。

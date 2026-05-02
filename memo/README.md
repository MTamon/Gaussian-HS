# memo/

このディレクトリは、CUDA 12.8 / PyTorch 2.9 / Python 3.11 / RTX 5090 (sm_120) 向け
改修にあたっての設計判断と、デモスクリプトを後から作成する際に必要な
共有情報を残しておくための場所。

ドキュメントの内容が古くなった場合は更新し、新しい設計判断が増えた場合は
ファイルを追加していくこと。

## ファイル一覧

| ファイル | 内容 |
| --- | --- |
| [pytorch3d_dual_mode.md](pytorch3d_dual_mode.md) | 内製コード / pytorch3d を環境変数で切り替える二系統運用の設計と、デモスクリプト作成時の env var スコーピング指針 |
| [environment_notes.md](environment_notes.md) | 目標環境（CUDA 12.8 / PyTorch 2.9 / RTX 5090 / gcc 11 / NumPy 2.2.6 / chumpy 0.71）と pip install --no-deps 運用、各依存の補足 |
| [code_change_log.md](code_change_log.md) | 既存の改修内容（torch.load wrapper、pytorch3d 内製化、torch.meshgrid indexing 等）と、今後デモスクリプト等で踏まえるべき点 |

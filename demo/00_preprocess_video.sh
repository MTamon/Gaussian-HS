#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo step 0: SMIRK + RVM + face-parsing + DWpose preprocessing from raw mp4.
#
# Builds an IMavatar-format dataset under
#   <repo>/../data/datasets/<subject>/<subject>/<split>/
# ready for `bash demo/02_train_subject.sh --subject <subject>` and the
# subsequent reenactment script.
#
# Wraps code/scripts/preprocess_smirk.py with conda-env / weight-existence
# checks and sane defaults so a typical run is just:
#   bash demo/00_preprocess_video.sh --video path/to/clip.mp4 --subject 999
#
# Usage:
#   bash demo/00_preprocess_video.sh --video PATH --subject NAME [options]
#   bash demo/00_preprocess_video.sh --help
#
# Required:
#   --video PATH           input mp4/mov
#   --subject NAME         dataset subject id (used as directory name)
#
# Optional:
#   --split SPLIT          train|test (default: train)
#   --image-size N         crop side in px           (default: 512)
#   --fps N                frame extraction fps      (default: 25)
#   --bb-scale F           stable-bbox scale         (default: 2.0)
#   --per-frame-bbox       use per-frame bbox (default: one fixed bbox/video)
#   --start-stage N        re-run pipeline from stage N (1..9)
#   --end-stage N          stop after stage N        (default: 9)
#   --data-root DIR        dataset root override     (default: <repo>/../data/datasets)
#   --smirk-repo DIR       MTamon/smirk checkout     (default: /home/mikawa/lab/outcome/smirk)
#   --hr-preprocess DIR   HRAvatar/preprocess root  (default: /home/mikawa/lab/outcome/HRAvatar/preprocess)
#   --smirk-batch-size N  SMIRK encoder batch size  (default: 16)
#   --help
# -----------------------------------------------------------------------------
set -eo pipefail

VIDEO=""
SUBJECT=""
SPLIT="train"
IMAGE_SIZE="512"
FPS="25"
BB_SCALE="2.0"
PER_FRAME_BBOX="0"
START_STAGE="1"
END_STAGE="9"
DATA_ROOT=""
SMIRK_REPO="/home/mikawa/lab/outcome/smirk"
HR_PREPROCESS="/home/mikawa/lab/outcome/HRAvatar/preprocess"
SMIRK_BATCH_SIZE="16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video) VIDEO="$2"; shift 2 ;;
        --subject) SUBJECT="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --image-size) IMAGE_SIZE="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --bb-scale) BB_SCALE="$2"; shift 2 ;;
        --per-frame-bbox) PER_FRAME_BBOX="1"; shift ;;
        --start-stage) START_STAGE="$2"; shift 2 ;;
        --end-stage) END_STAGE="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --smirk-repo) SMIRK_REPO="$2"; shift 2 ;;
        --hr-preprocess) HR_PREPROCESS="$2"; shift 2 ;;
        --smirk-batch-size) SMIRK_BATCH_SIZE="$2"; shift 2 ;;
        -h|--help)
            awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
            exit 0
            ;;
        *) echo "[00_preprocess_video.sh] unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${VIDEO}" || -z "${SUBJECT}" ]]; then
    echo "[00_preprocess_video.sh] --video and --subject are required" >&2
    exit 2
fi
if [[ ! -f "${VIDEO}" ]]; then
    echo "[00_preprocess_video.sh] video not found: ${VIDEO}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "gaussian-hs" ]]; then
    printf '\033[1;33m[warn] CONDA_DEFAULT_ENV=%s (expected gaussian-hs). Continuing.\033[0m\n' \
        "${CONDA_DEFAULT_ENV:-<unset>}"
fi

if [[ ! -d "${SMIRK_REPO}" ]]; then
    echo "[00_preprocess_video.sh] SMIRK repo not found: ${SMIRK_REPO}" >&2
    echo "  Clone https://github.com/MTamon/smirk (release/cuda128) or pass --smirk-repo." >&2
    exit 2
fi
SMIRK_CKPT="${SMIRK_REPO}/pretrained_models/SMIRK_em1.pt"
if [[ ! -f "${SMIRK_CKPT}" ]]; then
    echo "[00_preprocess_video.sh] SMIRK weights missing: ${SMIRK_CKPT}" >&2
    echo "  See ${SMIRK_REPO}/prepare_demos.sh or download_assets.sh." >&2
    exit 2
fi

if [[ ! -d "${HR_PREPROCESS}" ]]; then
    echo "[00_preprocess_video.sh] HRAvatar preprocess dir not found: ${HR_PREPROCESS}" >&2
    echo "  Clone HRAvatar at /home/mikawa/lab/outcome/HRAvatar or pass --hr-preprocess." >&2
    exit 2
fi
RVM_CKPT="${HR_PREPROCESS}/submodules/RobustVideoMatting/rvm_resnet50.pth"
PARSE_CKPT="${HR_PREPROCESS}/submodules/face-parsing.PyTorch/res/cp/79999_iter.pth"
if [[ ! -f "${RVM_CKPT}" ]]; then
    echo "[00_preprocess_video.sh] RVM weights missing: ${RVM_CKPT}" >&2
    echo "  Run HRAvatar/download_assets.sh." >&2
    exit 2
fi
if [[ ! -f "${PARSE_CKPT}" ]]; then
    echo "[00_preprocess_video.sh] face-parsing weights missing: ${PARSE_CKPT}" >&2
    echo "  Run HRAvatar/download_assets.sh." >&2
    exit 2
fi

PY_ARGS=(
    "${REPO_DIR}/code/scripts/preprocess_smirk.py"
    --video "${VIDEO}"
    --subject "${SUBJECT}"
    --split "${SPLIT}"
    --image-size "${IMAGE_SIZE}"
    --fps "${FPS}"
    --bb-scale "${BB_SCALE}"
    --start-stage "${START_STAGE}"
    --end-stage "${END_STAGE}"
    --smirk-repo "${SMIRK_REPO}"
    --hr-preprocess "${HR_PREPROCESS}"
    --smirk-batch-size "${SMIRK_BATCH_SIZE}"
)
if [[ -n "${DATA_ROOT}" ]]; then
    PY_ARGS+=(--data-root "${DATA_ROOT}")
fi
if [[ "${PER_FRAME_BBOX}" == "1" ]]; then
    PY_ARGS+=(--per-frame-bbox)
fi

echo "[00_preprocess_video.sh] video    = ${VIDEO}"
echo "[00_preprocess_video.sh] subject  = ${SUBJECT}/${SPLIT}"
echo "[00_preprocess_video.sh] stages   = ${START_STAGE}..${END_STAGE}"
echo "[00_preprocess_video.sh] smirk    = ${SMIRK_REPO}"
echo "[00_preprocess_video.sh] hr-pre   = ${HR_PREPROCESS}"

cd "${REPO_DIR}/code"
python "${PY_ARGS[@]}"

echo "[00_preprocess_video.sh] done. Inspect debug.mp4 under data/datasets/${SUBJECT}/${SUBJECT}/${SPLIT}/"

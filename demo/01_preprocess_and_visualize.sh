#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo step 1: dataset sanity-check + feature overlay visualization.
#
# Verifies that the prepared dataset under <repo>/../data/datasets/<subject>/
# is loadable, and renders per-frame composite overlays (RGB + mask + DWpose
# + FLAME 2D landmarks) so the user can confirm preprocessing alignment
# before training. Calls demo/visualize_features.py.
#
# Note: actual feature extraction from raw video (DWpose detection + FLAME
# tracking) is in the upstream TODO list and not in this repo. Running
# `bash download_assets.sh` is the closest the repo gets to "preprocessing".
#
# Usage:
#   bash demo/01_preprocess_and_visualize.sh
#   bash demo/01_preprocess_and_visualize.sh --split test --num-frames 30 --mp4
#   bash demo/01_preprocess_and_visualize.sh --pytorch3d
#   bash demo/01_preprocess_and_visualize.sh --help
#
# Flags:
#   --subject NAME     dataset subject (default: 001)
#   --split SPLIT      train|test (default: train)
#   --num-frames N     cap on number of frames to render (default: 60, 0=all)
#   --stride S         sample every Sth frame (default: 1)
#   --mp4              also write overlay.mp4 (10 fps, h264) under --out
#   --out DIR          output directory (default: demo/output/01_overlay/<subject>/<split>)
#   --data-root DIR    dataset root override (default: <repo>/../data/datasets)
#   --pytorch3d        use upstream PyTorch3D backend (scoped to this invocation)
#   --help             print this help
# -----------------------------------------------------------------------------
set -eo pipefail

SUBJECT="001"
SPLIT="train"
NUM_FRAMES="60"
STRIDE="1"
MP4="0"
OUT=""
DATA_ROOT=""
USE_P3D="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject) SUBJECT="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --num-frames) NUM_FRAMES="$2"; shift 2 ;;
        --stride) STRIDE="$2"; shift 2 ;;
        --mp4) MP4="1"; shift ;;
        --out) OUT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --pytorch3d) USE_P3D="1"; shift ;;
        -h|--help)
            awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
            exit 0
            ;;
        *) echo "[01_preprocess_and_visualize.sh] unknown flag: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${DATA_ROOT}" ]]; then
    DATA_ROOT="$(cd "${REPO_DIR}/.." && pwd)/data/datasets"
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "gaussian-hs" ]]; then
    printf '\033[1;33m[warn] CONDA_DEFAULT_ENV=%s (expected gaussian-hs). Continuing.\033[0m\n' \
        "${CONDA_DEFAULT_ENV:-<unset>}"
fi

INSTANCE_DIR="${DATA_ROOT}/${SUBJECT}/${SUBJECT}/${SPLIT}"
if [[ ! -d "${INSTANCE_DIR}" ]]; then
    echo "[01_preprocess_and_visualize.sh] dataset dir not found: ${INSTANCE_DIR}" >&2
    echo "  Run from repo root: bash download_assets.sh --flame_user USER --flame_pass PASS" >&2
    exit 2
fi
for required in flame_params.json image mask dwpose; do
    if [[ ! -e "${INSTANCE_DIR}/${required}" ]]; then
        echo "[01_preprocess_and_visualize.sh] missing ${INSTANCE_DIR}/${required}" >&2
        echo "  Re-run download_assets.sh; subject ${SUBJECT}/${SPLIT} is incomplete." >&2
        exit 2
    fi
done

if [[ -z "${OUT}" ]]; then
    OUT="${SCRIPT_DIR}/output/01_overlay/${SUBJECT}/${SPLIT}"
fi

PY_ARGS=(
    "${SCRIPT_DIR}/visualize_features.py"
    --data-root "${DATA_ROOT}"
    --subject   "${SUBJECT}"
    --split     "${SPLIT}"
    --num-frames "${NUM_FRAMES}"
    --stride     "${STRIDE}"
    --out        "${OUT}"
)
if [[ "${MP4}" == "1" ]]; then
    PY_ARGS+=(--mp4)
fi

echo "[01_preprocess_and_visualize.sh] data=${INSTANCE_DIR}"
echo "[01_preprocess_and_visualize.sh] out =${OUT}"
echo "[01_preprocess_and_visualize.sh] backend=$([[ "${USE_P3D}" == 1 ]] && echo pytorch3d || echo in-house)"

if [[ "${USE_P3D}" == "1" ]]; then
    GAUSSIAN_HS_USE_PYTORCH3D=1 python "${PY_ARGS[@]}"
else
    python "${PY_ARGS[@]}"
fi

echo "[01_preprocess_and_visualize.sh] done. Inspect overlays under ${OUT}"

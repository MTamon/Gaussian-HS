#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo step 2: train a per-subject ("personal adaptation") Gaussian-HS model.
#
# This is the only training mode this codebase supports — there is no
# pretraining / fine-tune split, every run trains a model from scratch on
# one subject. Mirrors the first python invocation in code/train.sh.
#
# Usage:
#   bash demo/02_train_subject.sh                                # subject 001, full
#   bash demo/02_train_subject.sh --quick                        # subset eval, fast turnaround
#   bash demo/02_train_subject.sh --epochs 5 --wandb-mode disabled
#   bash demo/02_train_subject.sh --pytorch3d
#   bash demo/02_train_subject.sh --help
#
# Flags:
#   --subject NAME       dataset subject (default: 001)
#   --conf PATH          conf path (default: code/configs/ghs.conf;
#                        accepts absolute or relative-to-repo or relative-to-code/)
#   --epochs N           --nepoch passed to exp_runner.py (default: 20)
#   --quick              add --quick_eval (subset for fast eval pass)
#   --wandb-mode MODE    online|offline|disabled (default: disabled)
#   --data-root DIR      dataset root (default: <repo>/../data/datasets)
#   --log-dir DIR        experiments root (default: <repo>/../log)
#   --pytorch3d          use upstream PyTorch3D backend (scoped to this invocation)
#   --help               print this help
# -----------------------------------------------------------------------------
set -eo pipefail

SUBJECT="001"
CONF="code/configs/ghs.conf"
EPOCHS="20"
QUICK="0"
WANDB_MODE="disabled"
DATA_ROOT=""
LOG_DIR=""
USE_P3D="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject) SUBJECT="$2"; shift 2 ;;
        --conf) CONF="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --quick) QUICK="1"; shift ;;
        --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --pytorch3d) USE_P3D="1"; shift ;;
        -h|--help)
            awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
            exit 0
            ;;
        *) echo "[02_train_subject.sh] unknown flag: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_DIR="${REPO_DIR}/code"
if [[ -z "${DATA_ROOT}" ]]; then
    DATA_ROOT="$(cd "${REPO_DIR}/.." && pwd)/data/datasets"
fi
DATA_DIR="${DATA_ROOT}"
if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="$(cd "${REPO_DIR}/.." && pwd)/log"
fi
mkdir -p "${LOG_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "gaussian-hs" ]]; then
    printf '\033[1;33m[warn] CONDA_DEFAULT_ENV=%s (expected gaussian-hs). Continuing.\033[0m\n' \
        "${CONDA_DEFAULT_ENV:-<unset>}"
fi

# Normalize CONF -> relative to code/
case "${CONF}" in
    /*) CONF_ABS="${CONF}" ;;
    *) CONF_ABS="$(cd "$(dirname "${CONF}")" 2>/dev/null && pwd)/$(basename "${CONF}")" || CONF_ABS="" ;;
esac
if [[ ! -f "${CONF_ABS}" ]]; then
    # Try: relative to repo root, then relative to code/
    if [[ -f "${REPO_DIR}/${CONF}" ]]; then
        CONF_ABS="${REPO_DIR}/${CONF}"
    elif [[ -f "${CODE_DIR}/${CONF}" ]]; then
        CONF_ABS="${CODE_DIR}/${CONF}"
    else
        echo "[02_train_subject.sh] cannot find conf: ${CONF}" >&2
        echo "  Tried absolute, ${REPO_DIR}/${CONF}, and ${CODE_DIR}/${CONF}" >&2
        exit 2
    fi
fi
case "${CONF_ABS}" in
    "${CODE_DIR}/"*) CONF_REL="${CONF_ABS#"${CODE_DIR}/"}" ;;
    *)
        echo "[02_train_subject.sh] conf must live under ${CODE_DIR}/ (got ${CONF_ABS})" >&2
        exit 2
        ;;
esac

CONF_STEM="$(basename "${CONF_REL}" .conf)"
CONF_PARENT="$(basename "$(dirname "${CONF_REL}")")"
if [[ "${CONF_PARENT}" == "confs" ]]; then
    METHOD="${CONF_STEM}"
else
    METHOD="${CONF_PARENT}/${CONF_STEM}"
fi
# train_split_name mirrors code/scripts/exp_runner.py:55,84 — subject 001
# overrides dataset.train.sub_dir=['train']; other subjects use the default
# ['all'] from code/configs/default.conf:76.
case "${SUBJECT}" in
    001) SPLIT="train" ;;
    *)   SPLIT="all" ;;
esac
EXPDIR="${LOG_DIR}/${SUBJECT}/${METHOD}/${SPLIT}/train"

# Verify dataset
INSTANCE_DIR="${DATA_DIR}/${SUBJECT}/${SUBJECT}/train"
if [[ ! -d "${INSTANCE_DIR}" ]]; then
    echo "[02_train_subject.sh] dataset not found: ${INSTANCE_DIR}" >&2
    echo "  Run from repo root: bash download_assets.sh --flame_user USER --flame_pass PASS" >&2
    exit 2
fi

# Verify FLAME assets (PointAvatar uses FLAME2020)
for asset in "${CODE_DIR}/flame/FLAME2020/generic_model.pkl" \
             "${CODE_DIR}/flame/FLAME2020/landmark_embedding.npy"; do
    if [[ ! -s "${asset}" ]]; then
        echo "[02_train_subject.sh] missing FLAME asset: ${asset}" >&2
        echo "  Run: bash download_assets.sh --flame_user USER --flame_pass PASS" >&2
        exit 2
    fi
done

# Generate a temp conf that includes the user's conf and absolutizes the data
# / log paths. The path is laid out as <tmp>/configs/<stem>.conf so that
# exp_runner.py's `Path(conf_path).parent.name` derives methodname=configs/<stem>
# (matching the on-disk convention from configs/ghs.conf).
TMP_CONF_DIR="${TMPDIR:-/tmp}/ghs-demo/configs"
mkdir -p "${TMP_CONF_DIR}"
TMP_CONF="${TMP_CONF_DIR}/${CONF_STEM}.conf"
cat > "${TMP_CONF}" <<EOF
include required("${CONF_ABS}")

train {
    exps_folder = "${LOG_DIR}/"
}
dataset {
    data_folder = "${DATA_DIR}"
}
EOF

EXTRA_ARGS=()
if [[ "${QUICK}" == "1" ]]; then
    EXTRA_ARGS+=(--quick_eval)
fi

echo "[02_train_subject.sh] subject=${SUBJECT}  conf=${CONF_REL}  method=${METHOD}"
echo "[02_train_subject.sh] data=${INSTANCE_DIR}"
echo "[02_train_subject.sh] log =${LOG_DIR}"
echo "[02_train_subject.sh] backend=$([[ "${USE_P3D}" == 1 ]] && echo pytorch3d || echo in-house)"
echo "[02_train_subject.sh] expected output: ${EXPDIR}/checkpoints/ModelParameters/latest.pth"

cd "${CODE_DIR}"

if [[ "${USE_P3D}" == "1" ]]; then
    GAUSSIAN_HS_USE_PYTORCH3D=1 python scripts/exp_runner.py \
        --conf "${TMP_CONF}" \
        --subject "${SUBJECT}" \
        --nepoch "${EPOCHS}" \
        --wandb_mode "${WANDB_MODE}" \
        "${EXTRA_ARGS[@]}"
else
    python scripts/exp_runner.py \
        --conf "${TMP_CONF}" \
        --subject "${SUBJECT}" \
        --nepoch "${EPOCHS}" \
        --wandb_mode "${WANDB_MODE}" \
        "${EXTRA_ARGS[@]}"
fi

echo "[02_train_subject.sh] done. Checkpoints under ${EXPDIR}/checkpoints/"

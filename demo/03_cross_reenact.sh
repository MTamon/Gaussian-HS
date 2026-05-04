#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Demo step 3: cross-reenactment with a trained Gaussian-HS model.
#
# Drives a model trained on --source with the FLAME / DWpose sequence of
# --target. If --target is omitted, falls back to self-reenactment using
# the source subject's own test split, which always runs after step 2 +
# `download_assets.sh` (no extra data needed).
#
# Usage:
#   bash demo/03_cross_reenact.sh                                # self-reenact 001 -> 001/test
#   bash demo/03_cross_reenact.sh --target Turnbull3             # requires data/datasets/Turnbull3/
#   bash demo/03_cross_reenact.sh --quick --fast-test            # subset eval + distilled-network path
#   bash demo/03_cross_reenact.sh --conf-reenact code/configs/reenact_002.conf
#   bash demo/03_cross_reenact.sh --help
#
# Flags:
#   --source NAME        trained source subject (default: 001)
#   --target NAME        driving subject (default: same as --source -> self-reenact)
#   --conf PATH          source-model conf path (default: code/configs/ghs.conf)
#   --conf-reenact PATH  reenact overlay conf (default: auto-generated for --target)
#   --max-frames N       cap on test.frame_interval upper bound (default: 300)
#   --quick              add --quick_eval
#   --fast-test          add --run_fast_test (distilled-network reenact path)
#   --data-root DIR      dataset root (default: <repo>/../data/datasets)
#   --log-dir DIR        experiments root (default: <repo>/../log)
#   --pytorch3d          use upstream PyTorch3D backend (scoped to this invocation)
#   --help               print this help
# -----------------------------------------------------------------------------
set -eo pipefail

SOURCE="001"
TARGET=""
CONF="code/configs/ghs.conf"
CONF_REENACT=""
MAX_FRAMES="300"
QUICK="0"
FAST_TEST="0"
DATA_ROOT=""
LOG_DIR=""
USE_P3D="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SOURCE="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --conf) CONF="$2"; shift 2 ;;
        --conf-reenact) CONF_REENACT="$2"; shift 2 ;;
        --max-frames) MAX_FRAMES="$2"; shift 2 ;;
        --quick) QUICK="1"; shift ;;
        --fast-test) FAST_TEST="1"; shift ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --pytorch3d) USE_P3D="1"; shift ;;
        -h|--help)
            awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
            exit 0
            ;;
        *) echo "[03_cross_reenact.sh] unknown flag: $1" >&2; exit 2 ;;
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

if [[ "${CONDA_DEFAULT_ENV:-}" != "gaussian-hs" ]]; then
    printf '\033[1;33m[warn] CONDA_DEFAULT_ENV=%s (expected gaussian-hs). Continuing.\033[0m\n' \
        "${CONDA_DEFAULT_ENV:-<unset>}"
fi

if [[ -z "${TARGET}" ]]; then
    TARGET="${SOURCE}"
    echo "[03_cross_reenact.sh] no --target given; self-reenacting (${SOURCE} drives ${SOURCE}/test)"
fi

# Normalize CONF -> relative to code/
case "${CONF}" in
    /*) CONF_ABS="${CONF}" ;;
    *) CONF_ABS="$(cd "$(dirname "${CONF}")" 2>/dev/null && pwd)/$(basename "${CONF}")" || CONF_ABS="" ;;
esac
if [[ ! -f "${CONF_ABS}" ]]; then
    if [[ -f "${REPO_DIR}/${CONF}" ]]; then
        CONF_ABS="${REPO_DIR}/${CONF}"
    elif [[ -f "${CODE_DIR}/${CONF}" ]]; then
        CONF_ABS="${CODE_DIR}/${CONF}"
    else
        echo "[03_cross_reenact.sh] cannot find conf: ${CONF}" >&2
        exit 2
    fi
fi
case "${CONF_ABS}" in
    "${CODE_DIR}/"*) CONF_REL="${CONF_ABS#"${CODE_DIR}/"}" ;;
    *)
        echo "[03_cross_reenact.sh] conf must live under ${CODE_DIR}/ (got ${CONF_ABS})" >&2
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
EXPDIR="${LOG_DIR}/${SOURCE}/${METHOD}"
# train_split_name mirrors code/scripts/exp_runner.py:55,84 — subject 001
# overrides dataset.train.sub_dir=['train']; other subjects use the default
# ['all'] from code/configs/default.conf:76.
case "${SOURCE}" in
    001) SPLIT="train" ;;
    *)   SPLIT="all" ;;
esac
CKPT="${EXPDIR}/${SPLIT}/train/checkpoints/ModelParameters/latest.pth"
if [[ ! -f "${CKPT}" ]]; then
    echo "[03_cross_reenact.sh] no trained checkpoint at ${CKPT}" >&2
    echo "  Run: bash demo/02_train_subject.sh --subject ${SOURCE}" >&2
    exit 2
fi

# Generate a base temp conf (mirrors 02): include the user's conf and absolutize
# data / log paths so train.py / reenact.py see the same canonical absolute
# locations regardless of cwd.
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

# Verify driving-subject data
TARGET_DIR="${DATA_DIR}/${TARGET}/${TARGET}/test"
for required in flame_params.json dwpose; do
    if [[ ! -e "${TARGET_DIR}/${required}" ]]; then
        echo "[03_cross_reenact.sh] missing ${TARGET_DIR}/${required}" >&2
        if [[ "${TARGET}" == "${SOURCE}" ]]; then
            echo "  Source subject ${SOURCE} test split is incomplete; re-run download_assets.sh." >&2
        else
            echo "  Target subject ${TARGET} not present. Place its prepared dataset under ${DATA_DIR}/${TARGET}/${TARGET}/test/" >&2
        fi
        exit 2
    fi
done

# Resolve / generate the reenact overlay conf
if [[ -n "${CONF_REENACT}" ]]; then
    case "${CONF_REENACT}" in
        /*) REENACT_ABS="${CONF_REENACT}" ;;
        *) REENACT_ABS="$(cd "$(dirname "${CONF_REENACT}")" 2>/dev/null && pwd)/$(basename "${CONF_REENACT}")" || REENACT_ABS="" ;;
    esac
    if [[ ! -f "${REENACT_ABS}" ]]; then
        if [[ -f "${REPO_DIR}/${CONF_REENACT}" ]]; then
            REENACT_ABS="${REPO_DIR}/${CONF_REENACT}"
        elif [[ -f "${CODE_DIR}/${CONF_REENACT}" ]]; then
            REENACT_ABS="${CODE_DIR}/${CONF_REENACT}"
        else
            echo "[03_cross_reenact.sh] cannot find --conf-reenact: ${CONF_REENACT}" >&2
            exit 2
        fi
    fi
    case "${REENACT_ABS}" in
        "${CODE_DIR}/"*) REENACT_REL="${REENACT_ABS#"${CODE_DIR}/"}" ;;
        *)
            echo "[03_cross_reenact.sh] --conf-reenact must live under ${CODE_DIR}/ (got ${REENACT_ABS})" >&2
            exit 2
            ;;
    esac
    echo "[03_cross_reenact.sh] using user-supplied reenact conf: ${REENACT_REL}"
else
    REENACT_TMP_DIR="${TMPDIR:-/tmp}/ghs-demo"
    mkdir -p "${REENACT_TMP_DIR}"
    REENACT_TMP="${REENACT_TMP_DIR}/reenact_${TARGET}.conf"
    cat > "${REENACT_TMP}" <<EOF
dataset {
    test_reenact_subject = "${TARGET}"
    test {
        frame_interval = [0, ${MAX_FRAMES}]
    }
}
EOF
    REENACT_REL="${REENACT_TMP}"
    echo "[03_cross_reenact.sh] generated reenact conf: ${REENACT_TMP}"
fi

EXTRA_ARGS=()
if [[ "${QUICK}" == "1" ]]; then
    EXTRA_ARGS+=(--quick_eval)
fi
if [[ "${FAST_TEST}" == "1" ]]; then
    EXTRA_ARGS+=(--run_fast_test)
fi

EVAL_NAME="eval"
if [[ "${QUICK}" == "1" ]]; then
    EVAL_NAME="eval_quick"
fi
EXPECTED_OUT="${EXPDIR}/${SPLIT}/train/${EVAL_NAME}_reenact_${TARGET}"

echo "[03_cross_reenact.sh] source=${SOURCE}  target=${TARGET}  method=${METHOD}"
echo "[03_cross_reenact.sh] checkpoint=${CKPT}"
echo "[03_cross_reenact.sh] backend=$([[ "${USE_P3D}" == 1 ]] && echo pytorch3d || echo in-house)"
echo "[03_cross_reenact.sh] expected output: ${EXPECTED_OUT}/test.mp4"

cd "${CODE_DIR}"

if [[ "${USE_P3D}" == "1" ]]; then
    GAUSSIAN_HS_USE_PYTORCH3D=1 python scripts/exp_runner.py \
        --conf "${TMP_CONF}" \
        --subject "${SOURCE}" \
        --is_reenact \
        --conf_reenact "${REENACT_REL}" \
        "${EXTRA_ARGS[@]}"
else
    python scripts/exp_runner.py \
        --conf "${TMP_CONF}" \
        --subject "${SOURCE}" \
        --is_reenact \
        --conf_reenact "${REENACT_REL}" \
        "${EXTRA_ARGS[@]}"
fi

echo "[03_cross_reenact.sh] done. Output under ${EXPECTED_OUT}/"

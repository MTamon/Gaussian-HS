#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Gaussian-HS third-party asset downloader.
#
# Downloads FLAME assets and the demo/training subject dataset expected by this repo.
#
# FLAME assets are placed at:
#   code/flame/FLAME2020/generic_model.pkl
#   code/flame/FLAME2020/landmark_embedding.npy
#   code/flame/FLAME2020/FLAME_masks.pkl
#   code/flame/FLAME2023/flame2023.pkl
#
# Dataset assets are placed at:
#   ../data/datasets/001/001/train
#   ../data/datasets/001/001/test
#
# Usage:
#   bash download_assets.sh
#   bash download_assets.sh --flame_user USER --flame_pass PASS
#   bash download_assets.sh --no_flame
#   bash download_assets.sh --no_dataset
#   bash download_assets.sh --data_root ../data/datasets
#   bash download_assets.sh --help
#
# FLAME2020/FLAME2023 are credential-gated through the FLAME download.php
# endpoint. FLAME_masks.zip is downloaded from the public MPI FLAME file server.
# Existing non-empty assets are not downloaded again.
#
# The subject 001 dataset is assembled from:
# - PointAvatar/IMavatar subject3.zip
# - Gaussian-HS release 001.zip, downloaded from the README Google Drive folder
# -----------------------------------------------------------------------------
set -euo pipefail

WITH_FLAME=1
WITH_DATASET=1
FLAME_USER=""
FLAME_PASS=""
DATA_ROOT="../data/datasets"
POINTAVATAR_SUBJECT3_URL="https://dataset.ait.ethz.ch/downloads/IMavatar_data/data/subject3.zip"
GHS_DATA_FOLDER_URL="https://drive.google.com/drive/folders/123DTRPc-Gfpl3pKbzmuNk4nBk72vWyS_?usp=sharing"
LANDMARK_EMBEDDING_URL="https://huggingface.co/Skywork/SkyReels-A1/resolve/e8f62f871898c2323750f26614086e52b6e1ea15/extra_models/FLAME/landmark_embedding.npy"
LANDMARK_EMBEDDING_SHA256="8095348eeafce5a02f6bd8765146307f9567a3f03b316d788a2e47336d667954"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no_flame) WITH_FLAME=0; shift ;;
        --no_dataset) WITH_DATASET=0; shift ;;
        --flame_user) FLAME_USER="$2"; shift 2 ;;
        --flame_pass) FLAME_PASS="$2"; shift 2 ;;
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        -h|--help)
            awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
            exit 0
            ;;
        *) echo "[download_assets.sh] unknown arg: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

say() { printf '\n\033[1;36m[download_assets.sh] %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m[download_assets.sh] WARN: %s\033[0m\n' "$*"; }

need_bin() {
    command -v "$1" >/dev/null 2>&1 || {
        warn "required binary '$1' is missing on PATH."
        return 1
    }
}

urle() {
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

install_first_match() {
    local root="$1"
    local pattern="$2"
    local dest="$3"
    local found
    found="$(find "${root}" -type f -name "${pattern}" | head -n 1 || true)"
    if [[ -n "${found}" ]]; then
        cp -f "${found}" "${dest}"
        say "Installed ${dest}"
    else
        warn "Could not find ${pattern} under ${root}"
        return 1
    fi
}

find_first_match() {
    local root="$1"
    local pattern="$2"
    find "${root}" -type f -name "${pattern}" | head -n 1 || true
}

valid_download() {
    local file="$1"
    local min_bytes="$2"
    [[ -s "${file}" ]] || return 1
    [[ "$(wc -c < "${file}")" -ge "${min_bytes}" ]] || return 1
    if file "${file}" | grep -qiE 'HTML|ASCII text'; then
        return 1
    fi
}

verify_sha256() {
    local file="$1"
    local expected="$2"
    local actual
    actual="$(sha256sum "${file}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]]
}

install_landmark_embedding() {
    local search_root="$1"
    local dest="$2"
    local found
    found="$(find_first_match "${search_root}" "landmark_embedding.npy")"
    if [[ -n "${found}" ]]; then
        cp -f "${found}" "${dest}"
        say "Installed ${dest}"
        return 0
    fi

    say "landmark_embedding.npy is not included in this FLAME2020.zip; downloading RingNet/FLAME PyTorch landmark embedding."
    wget "${LANDMARK_EMBEDDING_URL}" \
        -O "${dest}.tmp" --no-check-certificate --continue
    valid_download "${dest}.tmp" 30000 || {
        warn "landmark_embedding.npy download did not look valid."
        rm -f "${dest}.tmp"
        return 1
    }
    verify_sha256 "${dest}.tmp" "${LANDMARK_EMBEDDING_SHA256}" || {
        warn "landmark_embedding.npy checksum mismatch."
        rm -f "${dest}.tmp"
        return 1
    }
    mv -f "${dest}.tmp" "${dest}"
    say "Installed ${dest}"
}

dataset_ready() {
    local root="$1"
    [[ -s "${root}/001/001/train/flame_params.json" ]] || return 1
    [[ -d "${root}/001/001/train/image" ]] || return 1
    [[ -d "${root}/001/001/train/mask" ]] || return 1
    [[ -d "${root}/001/001/train/dwpose" ]] || return 1
}

copy_subject3_as_001() {
    local unzip_root="$1"
    local data_root="$2"
    local src=""

    if [[ -d "${unzip_root}/subject3/subject3" ]]; then
        src="${unzip_root}/subject3/subject3"
    else
        src="$(find "${unzip_root}" -type f -name 'flame_params.json' -print -quit | xargs -r dirname || true)"
    fi

    if [[ -z "${src}" || ! -d "${src}" ]]; then
        warn "Could not locate subject3 contents after unzip."
        return 1
    fi

    mkdir -p "${data_root}/001/001"
    cp -a "${src}/." "${data_root}/001/001/"
    say "Installed PointAvatar subject3 as ${data_root}/001/001"
}

merge_ghs_001_zip() {
    local zip_path="$1"
    local data_root="$2"
    local unzip_root="$3"
    local src=""

    unzip -t "${zip_path}" >/dev/null
    unzip -o "${zip_path}" -d "${unzip_root}" >/dev/null

    if [[ -d "${unzip_root}/001/001" ]]; then
        src="${unzip_root}/001/001"
    elif [[ -d "${unzip_root}/001" ]]; then
        src="${unzip_root}/001"
    else
        src="$(find "${unzip_root}" -type d -name dwpose -print -quit | xargs -r dirname || true)"
    fi

    if [[ -z "${src}" || ! -d "${src}" ]]; then
        warn "Could not locate Gaussian-HS 001.zip contents after unzip."
        return 1
    fi

    mkdir -p "${data_root}/001/001"
    cp -a "${src}/." "${data_root}/001/001/"
    say "Merged Gaussian-HS 001.zip into ${data_root}/001/001"
}

download_ghs_release_001_zip() {
    local out_dir="$1"
    local out_zip="$2"

    if [[ -n "${GHS_001_ZIP:-}" ]]; then
        cp "${GHS_001_ZIP}" "${out_zip}"
        return 0
    fi

    if [[ -n "${GHS_001_ZIP_URL:-}" ]]; then
        wget "${GHS_001_ZIP_URL}" -O "${out_zip}" --no-check-certificate --continue
        return 0
    fi

    if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("gdown") else 1)
PY
    then
        warn "Python package gdown is required to download the Google Drive folder."
        warn "Install via setup.sh, or set GHS_001_ZIP=/path/to/001.zip."
        return 1
    fi

    python -m gdown --folder "${GHS_DATA_FOLDER_URL}" -O "${out_dir}" --remaining-ok
    local found
    found="$(find "${out_dir}" -type f -name '001.zip' -print -quit || true)"
    if [[ -z "${found}" ]]; then
        warn "Could not find 001.zip in Gaussian-HS Google Drive folder."
        warn "Set GHS_001_ZIP=/path/to/001.zip or GHS_001_ZIP_URL=<direct-url> and rerun."
        return 1
    fi
    cp "${found}" "${out_zip}"
}

ensure_flame_credentials() {
    if [[ -z "${FLAME_USER}" ]]; then
        read -r -p "Username (FLAME): " FLAME_USER
    fi
    if [[ -z "${FLAME_PASS}" ]]; then
        read -r -s -p "Password (FLAME): " FLAME_PASS
        echo
    fi
}

download_flame_file() {
    local sfile="$1"
    local out="$2"
    local user_enc pass_enc
    ensure_flame_credentials
    user_enc="$(urle "${FLAME_USER}")"
    pass_enc="$(urle "${FLAME_PASS}")"
    wget --post-data "username=${user_enc}&password=${pass_enc}" \
        "https://download.is.tue.mpg.de/download.php?domain=flame&sfile=${sfile}&resume=1" \
        -O "${out}" --no-check-certificate --continue
}

need_bin wget || exit 2
need_bin unzip || exit 2
need_bin file || exit 2
need_bin sha256sum || exit 2

mkdir -p code/flame/FLAME2020 code/flame/FLAME2023

FLAME2020_MODEL="code/flame/FLAME2020/generic_model.pkl"
FLAME2020_LMK="code/flame/FLAME2020/landmark_embedding.npy"
FLAME2020_MASKS="code/flame/FLAME2020/FLAME_masks.pkl"
FLAME2023_MODEL="code/flame/FLAME2023/flame2023.pkl"

if [[ ${WITH_FLAME} -eq 1 ]]; then
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "${TMP_DIR}"' EXIT

    if [[ -s "${FLAME2020_MODEL}" ]]; then
        say "FLAME2020 generic_model.pkl already present, skipping."
    else
        say "Downloading FLAME2020.zip from the FLAME website."
        download_flame_file "FLAME2020.zip" "${TMP_DIR}/FLAME2020.zip"
        valid_download "${TMP_DIR}/FLAME2020.zip" 1048576 || {
            warn "FLAME2020.zip download did not look valid. Check credentials/license access."
            exit 1
        }
        unzip -t "${TMP_DIR}/FLAME2020.zip" >/dev/null
        unzip -o "${TMP_DIR}/FLAME2020.zip" -d "${TMP_DIR}/FLAME2020" >/dev/null
        install_first_match "${TMP_DIR}/FLAME2020" "generic_model.pkl" "${FLAME2020_MODEL}"
    fi

    if [[ -s "${FLAME2020_LMK}" ]]; then
        say "FLAME2020 landmark_embedding.npy already present, skipping."
    else
        install_landmark_embedding "${TMP_DIR:-.}" "${FLAME2020_LMK}"
    fi

    if [[ -s "${FLAME2023_MODEL}" ]]; then
        say "FLAME2023 model already present, skipping."
    else
        say "Downloading FLAME2023.zip from the FLAME website."
        download_flame_file "FLAME2023.zip" "${TMP_DIR}/FLAME2023.zip"
        valid_download "${TMP_DIR}/FLAME2023.zip" 1048576 || {
            warn "FLAME2023.zip download did not look valid. Check credentials/license access."
            exit 1
        }
        unzip -t "${TMP_DIR}/FLAME2023.zip" >/dev/null
        unzip -o "${TMP_DIR}/FLAME2023.zip" -d "${TMP_DIR}/FLAME2023" >/dev/null
        install_first_match "${TMP_DIR}/FLAME2023" "flame2023.pkl" "${FLAME2023_MODEL}"
    fi

    if [[ -s "${FLAME2020_MASKS}" ]]; then
        say "FLAME_masks.pkl already present, skipping."
    else
        say "Downloading FLAME_masks.zip from the public MPI FLAME file server."
        wget "https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip" \
            -O "${TMP_DIR}/FLAME_masks.zip" --no-check-certificate --continue
        valid_download "${TMP_DIR}/FLAME_masks.zip" 1024 || {
            warn "FLAME_masks.zip download did not look valid."
            exit 1
        }
        unzip -t "${TMP_DIR}/FLAME_masks.zip" >/dev/null
        unzip -o "${TMP_DIR}/FLAME_masks.zip" -d "${TMP_DIR}/FLAME_masks" >/dev/null
        install_first_match "${TMP_DIR}/FLAME_masks" "FLAME_masks.pkl" "${FLAME2020_MASKS}"
    fi
else
    say "--no_flame given; skipping FLAME downloads."
fi

say "Asset summary:"
for f in \
    "${FLAME2020_MODEL}" \
    "${FLAME2020_LMK}" \
    "${FLAME2020_MASKS}" \
    "${FLAME2023_MODEL}"; do
    if [[ -f "${f}" ]]; then
        printf '  %-48s %s\n' "${f}" "$(du -h "${f}" | cut -f1)"
    else
        printf '  %-48s %s\n' "${f}" "MISSING"
    fi
done

if [[ ! -f "${FLAME2020_MODEL}" || ! -f "${FLAME2020_LMK}" ]]; then
    warn "Runtime FLAME2020 assets are missing. Place them under code/flame/FLAME2020/."
fi

case "${DATA_ROOT}" in
    /*) DATA_ROOT_ABS="${DATA_ROOT}" ;;
    *) DATA_ROOT_ABS="${SCRIPT_DIR}/${DATA_ROOT}" ;;
esac
if [[ ${WITH_DATASET} -eq 1 ]]; then
    mkdir -p "${DATA_ROOT}"
    DATA_ROOT_ABS="$(cd "${DATA_ROOT}" && pwd)"

    if dataset_ready "${DATA_ROOT_ABS}"; then
        say "Dataset 001 already present under ${DATA_ROOT_ABS}, skipping."
    else
        need_bin python || exit 2
        DATA_TMP_DIR="$(mktemp -d)"
        trap 'rm -rf "${TMP_DIR:-}" "${DATA_TMP_DIR:-}"' EXIT

        say "Downloading PointAvatar/IMavatar subject3.zip."
        wget "${POINTAVATAR_SUBJECT3_URL}" \
            -O "${DATA_TMP_DIR}/subject3.zip" --no-check-certificate --continue
        valid_download "${DATA_TMP_DIR}/subject3.zip" 1048576 || {
            warn "subject3.zip download did not look valid."
            exit 1
        }
        unzip -t "${DATA_TMP_DIR}/subject3.zip" >/dev/null
        unzip -o "${DATA_TMP_DIR}/subject3.zip" -d "${DATA_TMP_DIR}/subject3" >/dev/null
        copy_subject3_as_001 "${DATA_TMP_DIR}/subject3" "${DATA_ROOT_ABS}"

        say "Downloading Gaussian-HS release 001.zip from Google Drive."
        download_ghs_release_001_zip "${DATA_TMP_DIR}/ghs_release" "${DATA_TMP_DIR}/001.zip"
        valid_download "${DATA_TMP_DIR}/001.zip" 1024 || {
            warn "001.zip download did not look valid."
            exit 1
        }
        merge_ghs_001_zip "${DATA_TMP_DIR}/001.zip" "${DATA_ROOT_ABS}" "${DATA_TMP_DIR}/001"

        dataset_ready "${DATA_ROOT_ABS}" || {
            warn "Dataset 001 is still incomplete after merge."
            exit 1
        }
    fi
else
    say "--no_dataset given; skipping subject 001 dataset download."
fi

say "Dataset summary:"
DATASET_001="${DATA_ROOT_ABS}/001/001"
for p in \
    "${DATASET_001}/train/flame_params.json" \
    "${DATASET_001}/train/image" \
    "${DATASET_001}/train/mask" \
    "${DATASET_001}/train/dwpose" \
    "${DATASET_001}/test/flame_params.json" \
    "${DATASET_001}/test/image" \
    "${DATASET_001}/test/mask" \
    "${DATASET_001}/test/dwpose"; do
    if [[ -e "${p}" ]]; then
        printf '  %-64s %s\n' "${p}" "OK"
    else
        printf '  %-64s %s\n' "${p}" "MISSING"
    fi
done

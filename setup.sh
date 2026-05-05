#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Gaussian-HS deterministic installer for Python 3.11 + PyTorch 2.9.1 + CUDA 12.8.
# -----------------------------------------------------------------------------
# This follows the MTamon/GaussianAvatars cuda128 setup pattern:
# - use system CUDA 12.8, not CONDA_PREFIX, for CUDA_HOME
# - install exact pins with --no-deps to avoid resolver drift
# - install chumpy from mattloper/chumpy at a fixed SHA and verify version 0.71
# - initialise only missing submodules, leaving existing checkouts untouched
#
# Usage:
#   bash setup.sh              create conda env gaussian-hs, install deps/assets
#   bash setup.sh --pip-only   install into the active Python environment
#   bash setup.sh --no-assets  skip download_assets.sh at the end
#   bash setup.sh --help       show this help
#
# Preconditions:
# - System CUDA Toolkit 12.8 is installed at /usr/local/cuda-12.8 or CUDA_HOME.
# - gcc-11 / g++-11 are installed.
# -----------------------------------------------------------------------------
set -eo pipefail

PIP_ONLY=0
NO_ASSETS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pip-only) PIP_ONLY=1; shift ;;
        --no-assets) NO_ASSETS=1; shift ;;
        -h|--help)
            awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
            exit 0
            ;;
        *) echo "[setup.sh] unknown flag: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CC="${CC:-gcc-11}"
export CXX="${CXX:-g++-11}"

if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d "/usr/local/cuda-12.8" ]]; then
        export CUDA_HOME="/usr/local/cuda-12.8"
    elif [[ -d "/usr/local/cuda" ]]; then
        export CUDA_HOME="/usr/local/cuda"
    else
        echo "[setup.sh] CUDA_HOME is not set and /usr/local/cuda-12.8 was not found." >&2
        echo "[setup.sh] Set CUDA_HOME to your system CUDA 12.8 install path and rerun." >&2
        exit 1
    fi
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;12.0}"
export FORCE_CUDA=1

echo "[setup.sh] CC=${CC} CXX=${CXX}"
echo "[setup.sh] CUDA_HOME=${CUDA_HOME}"
echo "[setup.sh] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
nvcc --version || { echo "[setup.sh] nvcc not found on PATH" >&2; exit 1; }

# Conservative submodule auto-init. Existing checkouts are not modified.
if [[ -d .git && -f .gitmodules ]]; then
    for sm in diff-gaussian-rasterization simple-knn; do
        sm_path="submodules/${sm}"
        if [[ ! -e "${sm_path}/.git" ]]; then
            echo "[setup.sh] auto-initialising ${sm_path}"
            git submodule update --init "${sm_path}"
        fi
    done

    DGR_GLM_HEADER="submodules/diff-gaussian-rasterization/third_party/glm/glm/glm.hpp"
    if [[ -e "submodules/diff-gaussian-rasterization/.git" && ! -f "${DGR_GLM_HEADER}" ]]; then
        echo "[setup.sh] glm header missing under diff-gaussian-rasterization; initialising nested submodules."
        git -C submodules/diff-gaussian-rasterization submodule update --init --recursive
    fi
fi

if [[ ${PIP_ONLY} -eq 0 ]]; then
    echo "[1/5] Creating conda env gaussian-hs (Python 3.11)"
    if conda env list | awk '{print $1}' | grep -qx "gaussian-hs"; then
        echo " -> conda env 'gaussian-hs' already exists, skipping creation."
    else
        conda env create --file environment.yml
    fi
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate gaussian-hs
else
    echo "[1/5] Using currently-active python (pip-only mode)"
fi

install_no_deps() {
    python -m pip install --no-deps "$@"
}

echo "[2/5] Installing pinned Python dependencies"
python -m pip install --upgrade pip==25.2 setuptools==80.9.0 wheel==0.45.1

# Core torch stack.
install_no_deps filelock==3.20.0
install_no_deps fsspec==2025.10.0
install_no_deps jinja2==3.1.6
install_no_deps markupsafe==3.0.3
install_no_deps mpmath==1.3.0
install_no_deps networkx==3.5
install_no_deps sympy==1.14.0
install_no_deps typing_extensions==4.15.0
install_no_deps triton==3.5.1
install_no_deps nvidia-cublas-cu12==12.8.4.1
install_no_deps nvidia-cuda-cupti-cu12==12.8.90
install_no_deps nvidia-cuda-nvrtc-cu12==12.8.93
install_no_deps nvidia-cuda-runtime-cu12==12.8.90
install_no_deps nvidia-cudnn-cu12==9.10.2.21
install_no_deps nvidia-cufft-cu12==11.3.3.83
install_no_deps nvidia-cufile-cu12==1.13.1.3
install_no_deps nvidia-curand-cu12==10.3.9.90
install_no_deps nvidia-cusolver-cu12==11.7.3.90
install_no_deps nvidia-cusparse-cu12==12.5.8.93
install_no_deps nvidia-cusparselt-cu12==0.7.1
install_no_deps nvidia-nccl-cu12==2.27.5
install_no_deps nvidia-nvjitlink-cu12==12.8.93
install_no_deps nvidia-nvshmem-cu12==3.3.20
install_no_deps nvidia-nvtx-cu12==12.8.90
install_no_deps torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
install_no_deps torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Numeric / image / plotting stack used by code/.
install_no_deps numpy==2.2.6
install_no_deps scipy==1.16.3
install_no_deps pillow==12.0.0
install_no_deps imageio==2.37.2
install_no_deps tifffile==2025.10.16
install_no_deps lazy_loader==0.4
install_no_deps scikit-image==0.25.2
install_no_deps opencv-python==4.12.0.88
install_no_deps av==12.3.0
install_no_deps contourpy==1.3.3
install_no_deps cycler==0.12.1
install_no_deps fonttools==4.60.1
install_no_deps kiwisolver==1.4.9
install_no_deps matplotlib==3.10.7
install_no_deps packaging==25.0
install_no_deps pyparsing==3.2.5
install_no_deps python-dateutil==2.9.0.post0
install_no_deps six==1.17.0
install_no_deps trimesh==4.4.9

# Project runtime dependencies.
install_no_deps pyhocon==0.3.59
install_no_deps tqdm==4.67.1
install_no_deps pandas==2.3.3
install_no_deps pytz==2025.2
install_no_deps tzdata==2025.2
install_no_deps plyfile==1.1.2
install_no_deps einops==0.8.1
install_no_deps lpips==0.1.4
install_no_deps protobuf==4.25.5
install_no_deps PyYAML==6.0.3
install_no_deps requests==2.32.3
install_no_deps certifi==2024.8.30
install_no_deps charset-normalizer==3.4.0
install_no_deps idna==3.10
install_no_deps urllib3==2.2.3
install_no_deps beautifulsoup4==4.14.2
install_no_deps soupsieve==2.8
install_no_deps click==8.3.0
install_no_deps docker-pycreds==0.4.0
install_no_deps gdown==5.2.0
install_no_deps gitdb==4.0.12
install_no_deps gitpython==3.1.45
install_no_deps platformdirs==4.5.0
install_no_deps psutil==7.1.3
install_no_deps sentry-sdk==2.43.0
install_no_deps setproctitle==1.3.6
install_no_deps smmap==5.0.2
install_no_deps annotated_types==0.7.0
install_no_deps typing_inspection==0.4.2
install_no_deps pydantic_core==2.41.5
install_no_deps pydantic==2.12.4
install_no_deps wandb==0.22.3

# SMIRK / RVM / face-parsing / DWpose preprocessing dependencies.
# All pinned to the versions HRAvatar (cuda128) ships, so dependent
# subprocesses (RobustVideoMatting, face-parsing.PyTorch) and the SMIRK
# encoder behave identically here.
install_no_deps controlnet_aux==0.0.10
install_no_deps onnxruntime-gpu==1.20.1
install_no_deps mediapipe==0.10.21
install_no_deps timm==1.0.11
install_no_deps huggingface_hub==0.26.2
install_no_deps safetensors==0.4.5
install_no_deps loguru==0.7.2
install_no_deps attrs==24.2.0
install_no_deps flatbuffers==24.3.25
install_no_deps sounddevice==0.5.0
install_no_deps sentencepiece==0.2.0
install_no_deps coloredlogs==15.0.1
install_no_deps humanfriendly==10.0
install_no_deps pims==0.7
install_no_deps slicerator==1.1.0

# DWpose stack (yolox + dw-ll_ucoco_384, via controlnet_aux). Requires
# the mm-* family; mmcv has no prebuilt cu128/torch2.9 wheel and is built
# from source here. mmdet 3.3.0 caps mmcv at <2.2.0 so we pin mmcv 2.1.0.
# xtcocotools 1.14.3 wheel is built against numpy 1.x and breaks under
# numpy 2.2.6 with a dtype-size error; --no-binary forces a clean build.
install_no_deps openmim==0.3.9
install_no_deps mmengine==0.10.5
install_no_deps yapf==0.40.2 addict==2.4.0 termcolor==2.5.0
install_no_deps rich==13.9.4 markdown-it-py==3.0.0 mdurl==0.1.2 pygments==2.18.0
install_no_deps importlib_metadata==8.5.0 zipp==3.21.0
install_no_deps mmcv==2.1.0
install_no_deps mmdet==3.3.0
install_no_deps mmpose==1.3.2
install_no_deps terminaltables==3.1.10 shapely==2.0.6 munkres==1.1.4
install_no_deps json-tricks==3.17.3 pycocotools==2.0.8
python -m pip install --no-deps --no-binary xtcocotools xtcocotools==1.14.3

# Build helpers and local CUDA extensions.
install_no_deps ninja==1.13.0

CHUMPY_SHA="580566eafc9ac68b2614b64d6f7aaa84eebb70da"
install_no_deps "git+https://github.com/mattloper/chumpy.git@${CHUMPY_SHA}"
python - <<'PY'
import chumpy
assert chumpy.__version__ == "0.71", (
    f"chumpy version drift: got {chumpy.__version__}, expected 0.71"
)
assert hasattr(chumpy, "Ch"), "chumpy.Ch missing"
PY

echo "[3/5] Building local CUDA extensions"
if [[ ! -f submodules/diff-gaussian-rasterization/setup.py || ! -f submodules/simple-knn/setup.py ]]; then
    echo "[setup.sh] submodules are missing setup.py files." >&2
    echo "[setup.sh] Inspect submodules/* or run: git submodule update --init --recursive" >&2
    exit 1
fi
python -m pip install --no-build-isolation --no-deps ./submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation --no-deps ./submodules/simple-knn

echo "[4/5] Sanity check"
PYTHONPATH="${SCRIPT_DIR}/code:${PYTHONPATH:-}" python - <<'PY'
import torch, torchvision
print(torch.__version__, torchvision.__version__, torch.version.cuda)
print(torch.cuda.is_available())
import numpy
print(numpy.__version__)
import chumpy
assert chumpy.__version__ == "0.71"
import cv2, imageio, skimage, scipy, pandas, pyhocon, trimesh, lpips, wandb  # noqa: F401
import plyfile, einops  # noqa: F401
import gdown, bs4, soupsieve  # noqa: F401
import diff_gaussian_rasterization  # noqa: F401
import simple_knn  # noqa: F401
from model import pytorch3d_compat  # noqa: F401
import mediapipe, timm, controlnet_aux, onnxruntime  # noqa: F401  # preprocess deps
import mmcv, mmdet, mmpose  # noqa: F401  # DWpose stack
from mmcv.ops import nms  # noqa: F401  # mmcv CUDA build sanity
print("OK")
PY

echo "[5/5] pip check (informational)"
python -m pip check || true

if [[ ${NO_ASSETS} -eq 0 ]]; then
    echo "[opt] Downloading FLAME assets and subject 001 dataset (see download_assets.sh)"
    bash download_assets.sh
fi

echo "Done."

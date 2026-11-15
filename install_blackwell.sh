#!/usr/bin/env bash

# Install D3GA for NVIDIA Blackwell GPUs (for example, GeForce RTX 50 series).
# Run this from an activated Conda environment:
#   conda activate d3ga
#   ./install_blackwell.sh

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly EXPECTED_ENV="d3ga"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Error: no Conda environment is active." >&2
    echo "Run: conda activate ${EXPECTED_ENV}" >&2
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "${EXPECTED_ENV}" ]]; then
    echo "Error: activate the '${EXPECTED_ENV}' Conda environment first." >&2
    echo "Current environment: ${CONDA_DEFAULT_ENV:-unknown}" >&2
    exit 1
fi

if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
    echo "Error: this installer expects Python 3.10." >&2
    exit 1
fi

cd "${SCRIPT_DIR}"

echo "[1/8] Installing the CUDA 12.8 build toolchain"
conda install -y -c nvidia -c conda-forge \
    cuda-toolkit=12.8 "cmake<4" ninja libusb gcc_linux-64=12 gxx_linux-64=12

export CUDA_HOME="${CONDA_PREFIX}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="${CXX}"
# NVIDIA's Conda packages keep CUDA headers and libraries in a target-specific
# directory. PyTorch extension builds do not discover that layout on their own.
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_CUDA_ARCH_LIST="12.0"
export TCNN_CUDA_ARCHITECTURES="120"
export D3GA_CUDA_ARCH="120"
# PyTorch3D compilation uses substantial RAM per compiler process. A serial
# build is slower but avoids WSL terminating parallel compiler jobs.
export MAX_JOBS="${MAX_JOBS:-1}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "Error: nvcc was not installed at ${CUDA_HOME}/bin/nvcc." >&2
    exit 1
fi

echo "[2/8] Installing PyTorch 2.7.1 with CUDA 12.8"
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

echo "[3/8] Checking PyTorch and the GPU"
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check the WSL NVIDIA driver.")

major, minor = torch.cuda.get_device_capability()
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{major}{minor})")
if major < 12:
    raise SystemExit("This installer is intended for a Blackwell GPU (sm_120 or newer).")
PY

echo "[4/8] Installing Python dependencies"
python -m pip install -r requirements.txt
python -m pip install kornia lpips
python -m pip install --no-build-isolation chumpy==0.70

echo "[5/8] Installing PyTorch3D"
if python -c "import pytorch3d" 2>/dev/null; then
    echo "PyTorch3D is already installed; skipping its source build."
else
    python -m pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git"
fi

echo "[6/8] Installing tiny-cuda-nn"
if python -c "import tinycudann" 2>/dev/null; then
    echo "tiny-cuda-nn is already installed; skipping its source build."
else
    python -m pip install --no-build-isolation \
        "git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch"
fi

echo "[7/8] Building D3GA CUDA extensions for sm_120"
if python -c "import torch; import bvh_distance_queries_cuda, diff_gaussian_rasterization, simple_knn._C, sampling_points_cuda, sampling_segments_cuda" 2>/dev/null; then
    echo "D3GA CUDA extensions are already installed; skipping their source builds."
else
    python -m pip install --no-build-isolation ./submodules/bvh
    python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
    python -m pip install --no-build-isolation ./submodules/simple-knn
    python -m pip install --no-build-isolation ./submodules/tetra-sampler
fi

echo "[8/8] Building the tetrahedralization utility"
cmake --fresh -S submodules/tetrahedralize -B submodules/tetrahedralize/build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}"
cmake --build submodules/tetrahedralize/build --parallel "${MAX_JOBS}"

python - <<'PY'
import torch
import diff_gaussian_rasterization
import simple_knn._C
import bvh_distance_queries_cuda
import sampling_points_cuda
import sampling_segments_cuda

print(f"Installation complete. CUDA is available on {torch.cuda.get_device_name(0)}.")
PY

#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/submodules/tetra-sampler:${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export AIST_ROOT="${AIST_ROOT:-${HOME}/datasets/AIST++}"
export SMPL_MODEL="${SMPL_MODEL:-${HOME}/datasets/body_models/smpl/SMPL_NEUTRAL_NUMPY.pkl}"

CONDA_BASE="${CONDA_EXE%/bin/conda}"
if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="${HOME}/miniforge3"
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate d3ga
cd "${PROJECT_DIR}"
iteration="${1:-5000}"
if [[ $# -gt 0 ]]; then
    shift
fi
python test.py configs/aist_quick.yml "${iteration}" "$@"

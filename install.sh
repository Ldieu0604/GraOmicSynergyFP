#!/usr/bin/env bash
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Activate your conda Python 3.13 env first, for example: conda activate py313"
    exit 1
fi

CONDA_PYTHON="$CONDA_PREFIX/bin/python"

if [ ! -x "$CONDA_PYTHON" ]; then
    echo "Could not find python inside the active conda env: $CONDA_PREFIX"
    exit 1
fi

PYTORCH_VERSION="2.9.1"
PYTORCH_CUDA="cu130"
PYG_VERSION="2.7.0"
PYG_TORCH_WHL_VERSION="2.9.0"
PYG_WHL_URL="https://data.pyg.org/whl/torch-${PYG_TORCH_WHL_VERSION}+${PYTORCH_CUDA}.html"

# Use the conda env interpreter directly so an active .venv cannot shadow it.
"$CONDA_PYTHON" -c "import sys; assert sys.version_info[:2] == (3, 13), 'Active conda env is not Python 3.13: ' + sys.version"

# Install rdkit with conda to avoid pip wheel compatibility issues.
conda install -y -c conda-forge rdkit

# Upgrade pip before installing the Python package stack.
"$CONDA_PYTHON" -m pip install --upgrade pip

# Install the general scientific Python stack first.
"$CONDA_PYTHON" -m pip install --upgrade -r requirements.txt

# Install CUDA-enabled PyTorch from the official wheel index.
"$CONDA_PYTHON" -m pip install --upgrade \
    "torch==${PYTORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"

# Install the matching optional PyG CUDA extensions from the official wheel index.
"$CONDA_PYTHON" -m pip install --upgrade \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    -f "${PYG_WHL_URL}"

# torch_geometric is pure Python here; pin to the newest version your index exposes.
"$CONDA_PYTHON" -m pip install --upgrade "torch_geometric==${PYG_VERSION}"

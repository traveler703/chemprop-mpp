#!/bin/bash
# === Chemprop Workshop Environment Setup ===

set -e

ENV_NAME="chemprop-workshop"

echo "Creating conda environment '${ENV_NAME}' with Python 3.11..."
conda create -n ${ENV_NAME} python=3.11 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing Chemprop v2 and dependencies..."
pip install chemprop
pip install rdkit-pypi
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install jupyter ipywidgets
pip install python-pptx
pip install mols2grid  # optional, for nice molecular display

echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Verify with: python -c 'import chemprop; print(chemprop.__version__)'"

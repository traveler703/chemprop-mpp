@echo off
REM === Chemprop Workshop Environment Setup ===
REM Run this in Anaconda Prompt or with conda in PATH

echo Creating conda environment 'chemprop-workshop' with Python 3.11...
conda create -n chemprop-workshop python=3.11 -y

echo Activating environment...
call conda activate chemprop-workshop

echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Installing Chemprop v2 and dependencies...
pip install chemprop
pip install rdkit-pypi
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install jupyter ipywidgets
pip install python-pptx
pip install mols2grid  REM optional, for nice molecular display in notebook

echo.
echo === Setup Complete ===
echo Activate with: conda activate chemprop-workshop
echo Verify with: python -c "import chemprop; print(chemprop.__version__)"
pause

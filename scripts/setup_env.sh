#!/bin/bash
#SBATCH --job-name=sam2_setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
set -euo pipefail

# --- resolve per-user config -------------------------------------------------
: "${TRAINER_ENV:?TRAINER_ENV must be set (sbatch --export=ALL,TRAINER_ENV=/path/to/trainer.env)}"
# shellcheck source=/dev/null
source "$TRAINER_ENV"

# 1. Activate environment
cd "$PROJECT_DIR"
mkdir -p logs
source "$VENV_DIR/bin/activate"

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch with CUDA 12.1 (the bulk of the download)
echo "Downloading and installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Clone and install SAM 2
echo "Setting up SAM 2 repository..."
if [ ! -d "$INSTALL_ROOT/models/sam2" ]; then
    mkdir -p "$INSTALL_ROOT/models"
    cd "$INSTALL_ROOT/models"
    git clone https://github.com/facebookresearch/sam2.git
fi

echo "Installing SAM 2 package from $INSTALL_ROOT/models/sam2..."
cd "$INSTALL_ROOT/models/sam2"
pip install -e .

# 5. Install other requirements
echo "Installing other requirements..."
cd "$INSTALL_ROOT"
pip install -r requirements.txt

echo "Setup complete at $(date)"

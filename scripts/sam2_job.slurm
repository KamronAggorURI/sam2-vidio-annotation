#!/bin/bash
#SBATCH --job-name=sam2_shark_annotation
#SBATCH --output=logs/sam2_%j.out
#SBATCH --error=logs/sam2_%j.err
#SBATCH --partition=gpu              # Common name, check 'sinfo' for your cluster
#SBATCH --gpus=1                     # Request 1 GPU
#SBATCH --constraint="vram32"        # Request at least 32GB VRAM (Cluster specific)
#SBATCH --cpus-per-task=8            # 8 cores to prevent GPU bottlenecking
#SBATCH --mem=60G                    # 60GB System RAM to handle large image sets
#SBATCH --time=12:00:00              # 12 hour limit
set -euo pipefail

# --- resolve per-user config -------------------------------------------------
# Provided by `sbatch --export=ALL,TRAINER_ENV=/abs/path/to/trainer.env`.
: "${TRAINER_ENV:?TRAINER_ENV must be set (sbatch --export=ALL,TRAINER_ENV=/path/to/trainer.env)}"
# shellcheck source=/dev/null
source "$TRAINER_ENV"

module purge
# module load CUDA/12.1              # Load CUDA if required by your HPC

cd "$PROJECT_DIR"
mkdir -p logs "$RUNS_DIR"

# --- provenance: open the manifest, and always close it on exit --------------
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
LAUNCHED_VIA="${OCEANCTL_LAUNCHED_VIA:-manual}"
finish() {
    rc=$?
    python3 "$PROJECT_DIR/provenance.py" end --run-id "$RUN_ID" --runs-dir "$RUNS_DIR" \
        --status "$rc" || true
}
trap finish EXIT
python3 "$PROJECT_DIR/provenance.py" start --run-id "$RUN_ID" --runs-dir "$RUNS_DIR" \
    --job-type annotate --launched-via "$LAUNCHED_VIA" \
    --drive "${RCLONE_REMOTE}:${DRIVE_QUEUE}" || true

# --- sync queue from GDrive --------------------------------------------------
echo "Syncing queue from Google Drive..."
mkdir -p "$QUEUE_DIR"
timeout 120s rclone copy "${RCLONE_REMOTE}:${DRIVE_QUEUE}/" "$QUEUE_DIR" -v

# --- activate environment + run ----------------------------------------------
source "$VENV_DIR/bin/activate"
pip install -r requirements.txt --quiet   # keep environment in sync

echo "Starting SAM 2.1 Processing at $(date)"
python3 -u main.py
echo "Job finished at $(date)"

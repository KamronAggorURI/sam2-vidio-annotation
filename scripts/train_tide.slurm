#!/bin/bash
#SBATCH --job-name=train_tide_eval
#SBATCH --output=logs/train_tide_%j.out
#SBATCH --error=logs/train_tide_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

# --- resolve per-user config -------------------------------------------------
: "${TRAINER_ENV:?TRAINER_ENV must be set (sbatch --export=ALL,TRAINER_ENV=/path/to/trainer.env)}"
# shellcheck source=/dev/null
source "$TRAINER_ENV"

module purge
cd "$PROJECT_DIR"
mkdir -p logs "$RUNS_DIR"

# --- environment -------------------------------------------------------------
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install ultralytics roboflow tidecv omegaconf pyyaml --quiet

# --- provenance: open manifest, always finalize on exit ----------------------
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
LAUNCHED_VIA="${OCEANCTL_LAUNCHED_VIA:-manual}"
: "${RUN_CONFIG:=}"
TIDE_OUT=""
finish() {
    rc=$?
    python3 "$PROJECT_DIR/provenance.py" end --run-id "$RUN_ID" --runs-dir "$RUNS_DIR" \
        --status "$rc" ${TIDE_OUT:+--tide "$TIDE_OUT"} || true
}
trap finish EXIT
python3 "$PROJECT_DIR/provenance.py" start --run-id "$RUN_ID" --runs-dir "$RUNS_DIR" \
    --job-type train --launched-via "$LAUNCHED_VIA" ${RUN_CONFIG:+--config "$RUN_CONFIG"} || true

# --- train -------------------------------------------------------------------
echo "Starting Training at $(date)"
python3 train.py ${RUN_CONFIG:+--config "$RUN_CONFIG"}

# Locate the run just produced (train.py writes runs/last_run.txt).
RUN_NAME="$(cat "$RUNS_DIR/last_run.txt" 2>/dev/null || echo "")"
MODEL_PATH="$RUNS_DIR/$RUN_NAME/weights/best.pt"

# --- COCO ground truth for TIDE (download if missing) ------------------------
echo "Ensuring COCO annotations for TIDE..."
python3 - <<'PY'
import os
from roboflow import Roboflow
key = os.environ.get("ROBOFLOW_API_KEY")
if key and not os.path.exists("./dataset/coco"):
    rf = Roboflow(api_key=key)
    project = rf.workspace("DeepSeaObjectDetection").project("Fish-Detection-v5")
    project.version(1).download(model_format="coco", location="./dataset/coco")
PY

DATA_YAML="./dataset/data.yaml"
GT_JSON="./dataset/coco/valid/_annotations.coco.json"
TIDE_OUT="$PROJECT_DIR/tide_results_$(date +%Y%m%d_%H%M%S)"

# --- TIDE evaluation ---------------------------------------------------------
if [ -f "$MODEL_PATH" ]; then
    echo "Starting TIDE Evaluation at $(date)"
    python3 eval_tide.py \
        --model "$MODEL_PATH" \
        --data "$DATA_YAML" \
        --gt "$GT_JSON" \
        --out "$TIDE_OUT" \
        --type yolov11
else
    echo "ERROR: Model not found at $MODEL_PATH. Training might have failed." >&2
    exit 1
fi

echo "Job finished at $(date). TIDE results in $TIDE_OUT"

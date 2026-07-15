#!/bin/bash
# Sync the annotation queue from Google Drive on the login node, then submit the
# SLURM annotation job. All paths come from trainer.env (see trainer.env.example).
#
# Usage:
#   bash scripts/sync_and_run.sh              # sync + submit
#   bash scripts/sync_and_run.sh --no-submit  # sync only; submit the job by hand
set -euo pipefail

# --- resolve per-user config -------------------------------------------------
: "${TRAINER_ENV:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/trainer.env}"
if [ ! -f "$TRAINER_ENV" ]; then
    echo "ERROR: trainer.env not found at $TRAINER_ENV" >&2
    echo "       Copy trainer.env.example to trainer.env and edit it, or set TRAINER_ENV." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$TRAINER_ENV"

NO_SUBMIT=0
[ "${1:-}" = "--no-submit" ] && NO_SUBMIT=1

echo "1. Syncing queue from Google Drive (login node)..."
mkdir -p "$QUEUE_DIR"
if ! rclone copy "${RCLONE_REMOTE}:${DRIVE_QUEUE}/" "$QUEUE_DIR" -v; then
    echo "❌ Rclone sync failed. Job not submitted." >&2
    exit 1
fi
echo "✓ Sync complete."

if [ "$NO_SUBMIT" -eq 1 ]; then
    echo "--no-submit set; skipping submission. Submit by hand with:"
    echo "   sbatch --export=ALL,TRAINER_ENV=$TRAINER_ENV \"$PROJECT_DIR/scripts/sam2_job.slurm\""
    exit 0
fi

echo "2. Submitting SLURM job..."
cd "$PROJECT_DIR"
sbatch --export=ALL,TRAINER_ENV="$TRAINER_ENV" "$PROJECT_DIR/scripts/sam2_job.slurm"

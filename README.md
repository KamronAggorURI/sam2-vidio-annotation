# SAM 2 Video Annotation for Deep-Sea Shark Detection

**Automated annotation pipeline using SAM 2 to track sharks through underwater video footage, reducing manual annotation time from hours to minutes.**

## 📋 Overview

This Jupyter notebook workflow uses Meta's Segment Anything Model 2 (SAM 2) to automatically propagate annotations through video frames. Weakly-annotated frames (such as bounding boxes either by hand or generated via a detector) are used to prompt the segmentation model.


**Benefits:**
- Higher annotation quality for little cost
- Quick and easy
- Easily shareable

---

## 🚀 Quickstart

### Prerequisites
- Access to HPC cluster with GPU (tested on Unity @ URI)
- Slurm job scheduler
- Python 3.11+
- CUDA-compatible GPU (A100, L40S, or H100 recommended)

### Fastest path: `oceanctl`

Most people should use [`oceanctl`](https://github.com/KamronAggorURI/oceanctl), the
terminal setup + run manager. It installs rclone on your laptop and the Unity login
node, handles the Google OAuth for you, clones this repo, builds the environment,
downloads checkpoints, writes your `trainer.env`, and stages the Drive queue:

```sh
uv tool install oceanctl
oceanctl setup      # provision laptop + Unity
oceanctl stage      # sync the Drive queue and submit the annotation job
oceanctl run        # submit a configured training run
```

### Manual path

**1.** Ensure your annotated data lives in Google Drive under
`DeepSea_ObjectDetection/rclone/queue/`, and add a shortcut to `DeepSea_ObjectDetection`
at the root of your Drive. Your rclone remote MUST be named `gdrive` (read-only scope
is enough).

**2.** Copy `trainer.env.example` to `trainer.env` and edit the paths for your account
(or run `echo $HOME` on Unity and substitute). The scripts read every path from here —
you no longer edit `sync_and_run.sh` by hand. Clone this repo into a folder called
`trainer` (i.e. `$HOME/ocean_detect/trainer/sam2-vidio-annotation`).

**3.** Build the environment once: `sbatch --export=ALL,TRAINER_ENV=$PWD/trainer.env scripts/setup_env.sh`
then `bash scripts/checkpoint-download.sh`.

**4.** Stage + run: `bash scripts/sync_and_run.sh` (add `--no-submit` to stage only and
submit the SLURM job by hand — e.g. for teaching).

## Configuring training runs

`train.py` is config-driven. Precedence (low → high):
`configs/train_defaults.yaml` ← `RUN_CONFIG` (a run's YAML) ← CLI `key=value`:

```sh
python3 train.py --config configs/presets/class_aware_v7.yaml
python3 train.py --set fl_gamma=2.0 --set class_weighting=none
```

The Roboflow API key is read from `$ROBOFLOW_API_KEY` — set it in `trainer.env`, never
commit it.

## Provenance

Every run writes `runs/<run_id>/manifest.json` via `provenance.py` (called by the SLURM
scripts at job start/end, so hand-launched and OpenOnDemand runs are recorded too). A
manifest captures who/when, the Drive data used, Unity/GPU stats, the fully-resolved
config, and metrics parsed from Ultralytics `results.csv`. `oceanctl runs --pull`
mirrors them to your laptop for ablation comparison and CSV/markdown export. Nothing is
written back to Google Drive.


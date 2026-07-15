"""Underwater object-detection trainer (YOLOv11 / RT-DETR via Ultralytics).

Config-driven: every hyperparameter comes from the resolved OmegaConf
(configs/train_defaults.yaml <- RUN_CONFIG <- CLI key=value), so a run is fully
described by its config and reproducible from its provenance manifest.

    python3 train.py                          # defaults
    python3 train.py --config configs/presets/class_aware_v7.yaml
    python3 train.py --set fl_gamma=2.0 --set class_weighting=none
    RUN_CONFIG=/path/run.yaml python3 train.py   # e.g. from the SLURM job

The Roboflow API key is read from $ROBOFLOW_API_KEY (never hardcode it in the repo).
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import yaml
from omegaconf import DictConfig
from roboflow import Roboflow
from ultralytics import RTDETR, YOLO

import runconfig


# ───────────────────────── dataset ─────────────────────────
def download_dataset(cfg: DictConfig) -> Path:
    """Download the dataset from Roboflow in YOLOv11 format; return data.yaml path."""
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set. Export it (e.g. in trainer.env) before training."
        )
    dataset_dir = Path(cfg.dataset_dir)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(cfg.roboflow.workspace).project(cfg.roboflow.project)
    version = project.version(int(cfg.roboflow.version))
    version.download(model_format="yolov11", location=str(dataset_dir), overwrite=False)

    data_yaml = dataset_dir / "data.yaml"
    assert data_yaml.exists(), f"data.yaml not found at {data_yaml}"
    print(f"[dataset] Downloaded to: {dataset_dir}")
    return data_yaml


# ─────────────────────── class imbalance ───────────────────────
def compute_class_weights(cfg: DictConfig, data_yaml: Path) -> dict:
    """Per-class weights inversely proportional to frequency (min weight = 1.0).

    Only used when cfg.class_weighting == 'inverse_freq'. See the appendix at the
    bottom of this file for injecting hard per-class weights into the loss head.
    """
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    label_dir = Path(cfg.dataset_dir) / "train" / "labels"
    num_classes = data_cfg.get("nc", len(data_cfg.get("names", [])))

    counts: Counter = Counter()
    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1

    total = sum(counts.values()) or 1
    weights = {i: 1.0 / ((counts.get(i, 1) / total) * max(num_classes, 1))
               for i in range(num_classes)}
    min_w = min(weights.values()) if weights else 1.0
    weights = {k: v / min_w for k, v in weights.items()}

    names = data_cfg.get("names", {})
    print("[imbalance] Class weights (higher = rarer):")
    for idx, w in sorted(weights.items()):
        name = names[idx] if isinstance(names, list) else names.get(idx, str(idx))
        print(f"  [{idx}] {name:30s} count={counts.get(idx, 0):6d}  weight={w:.3f}")
    return weights


# ─────────────────────── training args ───────────────────────
def get_training_args(cfg: DictConfig, data_yaml: Path, run_name: str) -> dict:
    """Assemble the kwargs passed to model.train() from the resolved config."""
    args = {
        "data": str(data_yaml),
        "epochs": int(cfg.epochs),
        "imgsz": int(cfg.imgsz),
        "batch": int(cfg.batch),
        "workers": int(cfg.workers),
        "device": cfg.device,
        "project": str(cfg.runs_dir),
        "name": run_name,
        "exist_ok": True,
        # loss / imbalance
        "fl_gamma": float(cfg.fl_gamma),
        # optimizer
        "optimizer": str(cfg.optimizer),
        "lr0": float(cfg.lr0),
        "lrf": float(cfg.lrf),
        "warmup_epochs": int(cfg.warmup_epochs),
        "cos_lr": bool(cfg.cos_lr),
        "weight_decay": float(cfg.weight_decay),
        "dropout": float(cfg.dropout),
        # checkpointing / logging
        "save": True,
        "save_period": int(cfg.save_period),
        "patience": int(cfg.patience),
        "plots": bool(cfg.plots),
        "verbose": bool(cfg.verbose),
        # augmentation
        "fliplr": float(cfg.fliplr), "flipud": float(cfg.flipud),
        "degrees": float(cfg.degrees), "translate": float(cfg.translate),
        "scale": float(cfg.scale), "shear": float(cfg.shear),
        "hsv_h": float(cfg.hsv_h), "hsv_s": float(cfg.hsv_s), "hsv_v": float(cfg.hsv_v),
        "mosaic": float(cfg.mosaic), "mixup": float(cfg.mixup),
        "copy_paste": float(cfg.copy_paste), "blur": float(cfg.blur),
        "erasing": float(cfg.erasing),
    }
    if cfg.checkpoint and Path(str(cfg.checkpoint)).exists():
        print(f"[checkpoint] Resuming from: {cfg.checkpoint}")
        args["resume"] = True
    return args


def load_model(cfg: DictConfig):
    """Load YOLOv11 or RT-DETR from a checkpoint or size-appropriate pretrained weights."""
    if cfg.checkpoint and Path(str(cfg.checkpoint)).exists():
        ckpt = str(cfg.checkpoint)
        model = YOLO(ckpt) if cfg.model == "yolov11" else RTDETR(ckpt)
        print(f"[model] Loaded from checkpoint: {ckpt}")
        return model
    if cfg.model == "yolov11":
        weights = f"yolo11{cfg.model_size}-seg.pt"
        print(f"[model] YOLOv11 pretrained: {weights}")
        return YOLO(weights)
    size = cfg.model_size if cfg.model_size in ("l", "x") else "l"
    weights = f"rtdetr-{size}.pt"
    print(f"[model] RT-DETR pretrained: {weights}")
    return RTDETR(weights)


# ───────────────────────── train ─────────────────────────
def train(cfg: DictConfig):
    name = runconfig.run_name(cfg)
    print(f"\n{'='*50}\n Training: {cfg.model.upper()}  ({name})\n{'='*50}\n")

    data_yaml = download_dataset(cfg)
    if cfg.class_weighting == "inverse_freq":
        compute_class_weights(cfg, data_yaml)  # informs fl_gamma tuning; see appendix
    model = load_model(cfg)

    run_dir = Path(cfg.runs_dir) / name
    runconfig.save_resolved(cfg, run_dir / "resolved_config.yaml")
    # Pointer so downstream steps (TIDE eval, provenance) can find this run's outputs.
    Path(cfg.runs_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.runs_dir) / "last_run.txt").write_text(name + "\n")

    results = model.train(**get_training_args(cfg, data_yaml, name))
    print(f"\n[done] Training complete. Results in: {run_dir}")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Config-driven underwater detection trainer")
    p.add_argument("--config", help="run-config YAML (overrides train_defaults.yaml)")
    p.add_argument("--set", action="append", default=[], metavar="key=value",
                   help="override a config field (repeatable)")
    # Back-compat convenience flags → translated to overrides.
    p.add_argument("--model")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--checkpoint")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    dotlist = list(a.set)
    for key, val in (("model", a.model), ("epochs", a.epochs),
                     ("batch", a.batch), ("checkpoint", a.checkpoint)):
        if val is not None:
            dotlist.append(f"{key}={val}")
    cfg = runconfig.resolve(run_config=a.config, dotlist=dotlist)
    train(cfg)


if __name__ == "__main__":
    main()


# ───────────────────────── APPENDIX: hard per-class weights ─────────────────────────
# Focal loss (fl_gamma) is the simple, stable lever. For hard per-class loss weights,
# subclass the Ultralytics DetectionTrainer and inject a weights tensor into the loss
# head — see git history for the starter skeleton. This depends on Ultralytics internals
# and can break across versions, so keep it behind the class_weighting config switch.

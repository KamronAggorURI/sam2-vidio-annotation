"""Shared OmegaConf resolver for training runs.

Resolution precedence (low → high):
    configs/train_defaults.yaml  <-  RUN_CONFIG (a run's YAML)  <-  CLI key=value

Both train.py and provenance.py import this so the run manifest records exactly the
config that trained. RUN_CONFIG may be passed explicitly or via the environment
(the SLURM scripts export it).
"""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
DEFAULTS_PATH = REPO_ROOT / "configs" / "train_defaults.yaml"


def resolve(run_config: str | None = None,
            dotlist: list[str] | None = None) -> DictConfig:
    """Return the fully-resolved run config."""
    cfg = OmegaConf.load(DEFAULTS_PATH)
    rc = run_config or os.environ.get("RUN_CONFIG")
    if rc and Path(rc).exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(rc))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg  # type: ignore[return-value]


def run_name(cfg: DictConfig) -> str:
    """Deterministic run directory name for a config."""
    explicit = cfg.get("run_name")
    if explicit:
        return str(explicit)
    return f"{cfg.model}_{cfg.dataset}"


def save_resolved(cfg: DictConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(OmegaConf.to_yaml(cfg, sort_keys=True))

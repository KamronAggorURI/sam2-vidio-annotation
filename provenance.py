"""Run provenance: write a manifest at job start and finalize it at job end.

Called by the SLURM scripts (``provenance.py start`` / ``provenance.py end``), so a
manifest is produced for *every* run — whether launched by oceanctl, by hand, or via
OpenOnDemand. oceanctl later mirrors these manifests to the laptop for ablation
comparison. Writes only to the local ``runs/<run_id>/`` dir (never to Google Drive).

Manifest schema (v1) — see README "Provenance".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — provenance must never break the job
        return ""


def _manifest_path(runs_dir: str, run_id: str) -> Path:
    return Path(runs_dir) / run_id / "manifest.json"


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# environment / repo capture
# --------------------------------------------------------------------------- #
def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return _run(["git", "-C", str(cwd), *args])


def _gpu_info() -> tuple[str, int]:
    if not shutil.which("nvidia-smi"):
        return "", 0
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    gpus = [g.strip() for g in out.splitlines() if g.strip()]
    return (gpus[0] if gpus else ""), len(gpus)


def _versions() -> dict:
    info = {"python": sys.version.split()[0]}
    try:
        import torch  # noqa: PLC0415 — optional; present once the venv is active
        info["torch"] = torch.__version__
        info["cuda"] = getattr(torch.version, "cuda", None) or ""
    except Exception:  # noqa: BLE001
        pass
    if shutil.which("rclone"):
        v = _run(["rclone", "version"]).splitlines()
        if v:
            info["rclone"] = v[0].replace("rclone", "").strip()
    return info


def _capture_env() -> dict:
    gpu, gpu_count = _gpu_info()
    return {
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "gpu": gpu,
        "gpu_count": gpu_count or int(os.environ.get("SLURM_GPUS", 0) or 0),
        "cpus": os.environ.get("SLURM_CPUS_ON_NODE", ""),
        "mem": os.environ.get("SLURM_MEM_PER_NODE", ""),
        **_versions(),
    }


def _resolved_config(run_config: str | None) -> dict:
    """Resolve the training config the same way train.py does (best-effort)."""
    try:
        import runconfig  # noqa: PLC0415
        from omegaconf import OmegaConf  # noqa: PLC0415
        cfg = runconfig.resolve(run_config=run_config)
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return {}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _find_col(headers: list[str], *needles: str) -> str | None:
    for h in headers:
        low = h.lower()
        if all(n.lower() in low for n in needles):
            return h
    return None


def parse_results_csv(path: Path) -> dict:
    """Extract final + best detection metrics from an Ultralytics results.csv."""
    if not path.exists():
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    headers = [h.strip() for h in rows[0].keys()]
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]

    c_map50 = _find_col(headers, "map50", "(b)") or _find_col(headers, "map50")
    # mAP50-95 column contains "95"; plain mAP50 must not.
    c_map = _find_col(headers, "map50-95", "(b)") or _find_col(headers, "map", "95")
    c_p = _find_col(headers, "precision", "(b)") or _find_col(headers, "precision")
    c_r = _find_col(headers, "recall", "(b)") or _find_col(headers, "recall")

    def _f(row: dict, col: str | None):
        if not col or row.get(col) in (None, ""):
            return None
        try:
            return float(row[col])
        except ValueError:
            return None

    final = rows[-1]
    best_map = None
    if c_map:
        vals = [v for v in (_f(r, c_map) for r in rows) if v is not None]
        best_map = max(vals) if vals else None
    return {
        "map50": _f(final, c_map50),
        "map50_95": _f(final, c_map),
        "precision": _f(final, c_p),
        "recall": _f(final, c_r),
        "best_map50_95": best_map,
        "epochs": len(rows),
    }


def _run_name(runs_dir: str) -> str:
    pointer = Path(runs_dir) / "last_run.txt"
    if pointer.exists():
        return pointer.read_text().strip()
    return ""


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_start(a: argparse.Namespace) -> int:
    path = _manifest_path(a.runs_dir, a.run_id)
    manifest = {
        "schema": MANIFEST_VERSION,
        "run": {
            "id": a.run_id,
            "job_type": a.job_type,
            "launched_via": a.launched_via,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "submitter": os.environ.get("SLURM_JOB_USER") or os.environ.get("USER", ""),
            "submit_ts": os.environ.get("OCEANCTL_SUBMIT_TS", ""),
            "start_ts": _now(),
            "end_ts": "",
            "status": "running",
        },
        "repo": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "sam2_commit": _git("rev-parse", "HEAD", cwd=REPO_ROOT / "models" / "sam2"),
        },
        "data": {"drive_path": a.drive or "", "dataset": ""},
        "env": _capture_env(),
        "config": _resolved_config(a.config) if a.job_type == "train" else {},
        "metrics": {},
        "timing": {"start_ts": _now()},
        "artifacts": {},
    }
    if manifest["config"]:
        manifest["data"]["dataset"] = manifest["config"].get("dataset", "")
    _write(path, manifest)
    print(f"[provenance] started manifest at {path}")
    return 0


def cmd_end(a: argparse.Namespace) -> int:
    path = _manifest_path(a.runs_dir, a.run_id)
    manifest = _read(path)
    manifest.setdefault("run", {})
    manifest.setdefault("timing", {})
    manifest.setdefault("artifacts", {})

    ok = str(a.status) == "0"
    manifest["run"]["status"] = "completed" if ok else "failed"
    end_ts = _now()
    manifest["run"]["end_ts"] = end_ts
    manifest["timing"]["completed_ts"] = end_ts
    start_ts = manifest.get("timing", {}).get("start_ts")
    if start_ts:
        try:
            delta = datetime.fromisoformat(end_ts) - datetime.fromisoformat(start_ts)
            manifest["timing"]["wall_clock_s"] = int(delta.total_seconds())
        except ValueError:
            pass

    # env may be richer now that the venv is active (torch/cuda versions).
    manifest["env"] = {**manifest.get("env", {}), **_capture_env()}

    name = _run_name(a.runs_dir)
    if name:
        run_dir = Path(a.runs_dir) / name
        manifest["metrics"] = parse_results_csv(run_dir / "results.csv")
        best = run_dir / "weights" / "best.pt"
        manifest["artifacts"] = {
            "weights": str(best) if best.exists() else "",
            "results_csv": str(run_dir / "results.csv"),
            "plots_dir": str(run_dir),
            "tide": a.tide or "",
        }
    elif a.tide:
        manifest["artifacts"]["tide"] = a.tide

    _write(path, manifest)
    print(f"[provenance] finalized manifest ({manifest['run']['status']}) at {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Write/finalize a run provenance manifest")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("start")
    s.add_argument("--run-id", required=True)
    s.add_argument("--runs-dir", required=True)
    s.add_argument("--job-type", default="train", choices=["train", "annotate"])
    s.add_argument("--launched-via", default="manual")
    s.add_argument("--config", default=None)
    s.add_argument("--drive", default=None)
    s.set_defaults(func=cmd_start)

    e = sub.add_parser("end")
    e.add_argument("--run-id", required=True)
    e.add_argument("--runs-dir", required=True)
    e.add_argument("--status", required=True, help="exit code of the job step")
    e.add_argument("--tide", default=None)
    e.set_defaults(func=cmd_end)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

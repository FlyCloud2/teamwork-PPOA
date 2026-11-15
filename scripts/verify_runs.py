#!/usr/bin/env python
"""Verify run artifacts; --baseline requires the complete W2 matrix."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from run import REPO, inspect_progress

ENVS = ("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4")
BASELINE = {"total_timesteps": 1_000_000, "num_envs": 1, "num_steps": 2048,
            "num_minibatches": 32, "update_epochs": 10, "learning_rate": 3e-4,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2, "target_kl": None,
            "anneal_lr": True, "norm_adv": True, "clip_vloss": True, "ent_coef": 0.0,
            "vf_coef": 0.5, "max_grad_norm": 0.5, "hidden_sizes": [64, 64],
            "activation": "tanh", "adam_eps": 1e-5, "cuda": False, "torch_deterministic": True}


def verify_run(path, root=REPO):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("status") != "completed" or config.get("return_code") != 0:
        raise ValueError(f"{path.name}: not completed successfully")
    if config["run_id"] != path.name:
        raise ValueError("run directory and configuration ID differ")
    hp = config["hyperparameters"]
    if hp["batch_size"] != hp["num_steps"] * hp["num_envs"]:
        raise ValueError("invalid derived batch size")
    if hp["num_updates"] != math.ceil(hp["total_timesteps"] / hp["batch_size"]):
        raise ValueError("invalid derived update count")
    if (hp["env_id"], hp["seed"], hp["total_timesteps"]) != (config["env"], config["seed"], config["steps"]):
        raise ValueError("top-level configuration and training arguments differ")
    measured = inspect_progress(path / "progress.csv", hp)
    if any(config.get(key) != value for key, value in measured.items()):
        raise ValueError("recorded step/update totals differ from progress.csv")
    log = (path / "stdout.log").read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last)" in log or "SPS:" not in log:
        raise ValueError("stdout log missing successful training evidence")
    with (root / "logs" / "runs.csv").open(encoding="utf-8", newline="") as handle:
        entries = [row for row in csv.DictReader(handle) if row["run_id"] == path.name]
    if len(entries) != 1:
        raise ValueError("run must have exactly one index entry")
    for key in ("env", "seed", "variant", "steps", "actual_steps", "updates", "return_code", "status", "git_commit_short"):
        if entries[0][key] != str(config[key]):
            raise ValueError(f"index disagrees with config: {key}")
    return config


def verify_baseline(root=REPO, run_ids=None):
    if run_ids is None:
        manifest = json.loads((root / "logs" / "w2_baseline.json").read_text(encoding="utf-8"))
        run_ids = manifest["run_ids"]
    if len(run_ids) != 3 or len(set(run_ids)) != 3:
        raise ValueError("W2 requires exactly three distinct runs")
    configs = [verify_run(root / "runs" / rid, root) for rid in run_ids]
    if {config["env"] for config in configs} != set(ENVS):
        raise ValueError("W2 environment matrix is incomplete")
    for config in configs:
        if config["seed"] != 0 or config["variant"] != "clip0.2" or config["smoke"]:
            raise ValueError("W2 requires seed 0, clip0.2, non-smoke runs")
        for key, value in BASELINE.items():
            if config["hyperparameters"].get(key) != value:
                raise ValueError(f"baseline mismatch: {key}")
        if config["packages"].get("gymnasium") != "0.29.1" or config["packages"].get("mujoco") != "3.1.6":
            raise ValueError("baseline simulator versions differ from frozen versions")
        for name, digest in config["source_sha256"].items():
            if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"baseline source differs from current file: {name}")
        with (root / "runs" / config["run_id"] / "progress.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for key in ("episode_return", "episode_length", "explained_variance"):
                    if not math.isfinite(float(row[key])):
                        raise ValueError(f"baseline has non-finite {key} at step {row['global_step']}")
    if len({config["threads"] for config in configs}) != 1:
        raise ValueError("baseline thread counts differ")
    if any(config["packages"] != configs[0]["packages"] for config in configs):
        raise ValueError("baseline package versions differ")
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--root", type=Path, default=REPO)
    args = parser.parse_args()
    if not args.run and not args.baseline:
        parser.error("choose --run or --baseline")
    try:
        configs = verify_baseline(args.root) if args.baseline else [verify_run(args.run, args.root)]
    except (OSError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}")
        return 1
    for config in configs:
        print(f"PASS {config['run_id']}: {config['actual_steps']:,} steps, {config['updates']} updates")
    return 0


if __name__ == "__main__":
    sys.exit(main())

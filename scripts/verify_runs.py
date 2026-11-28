#!/usr/bin/env python
"""Verify run artifacts; --baseline requires the complete W2 matrix and --c2 the W4 one."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from run import REPO, inspect_progress

ENVS = ("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4")
SEEDS = (0, 1, 2)
BASELINE = {"total_timesteps": 1_000_000, "num_envs": 1, "num_steps": 2048,
            "num_minibatches": 32, "update_epochs": 10, "learning_rate": 3e-4,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_coef": 0.2, "target_kl": None,
            "anneal_lr": True, "norm_adv": True, "clip_vloss": True, "ent_coef": 0.0,
            "vf_coef": 0.5, "max_grad_norm": 0.5, "hidden_sizes": [64, 64],
            "activation": "tanh", "adam_eps": 1e-5, "cuda": False, "torch_deterministic": True}
# [REPRO] W4 / C2：只允许 K 这一个因子在两个臂之间变化。
C2_EPOCHS = (1, 10)
C2_VARIANT = "clip0.2"
C2_STEPS = 1_000_000
# Per-run bookkeeping that is *not* an experimental factor: the environment/seed
# identify the cell, and the output paths necessarily encode the run directory.
# Everything else in `hyperparameters` must be identical between the two arms.
C2_NOT_A_FACTOR = ("update_epochs", "env_id", "seed", "csv_out", "tb_out")
# [REPRO] W5 / extension：同一个矩阵，只把 K（update_epochs）换成扫描集合。
# K = 1 与 K = 10 复用 W4/C2 的 run，因此这两项必须与 C2_EPOCHS 保持一致。
KSCAN_EPOCHS = (1, 3, 10, 20)
KSCAN_VARIANT = C2_VARIANT
KSCAN_STEPS = C2_STEPS


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
    # The executed source is identified by the hashes in each run's config.json.
    # Later weeks legitimately edit shared files (W3 added the KL-penalty flags to
    # src/ppo_args.py), so drift against the current worktree is reported, not
    # failed: that comparison detects an unnoticed substitution, not evolution.
    hashes = [config["source_sha256"] for config in configs]
    if any(digests != hashes[0] for digests in hashes[1:]):
        raise ValueError("the three W2 runs executed different training sources")
    drifted = sorted(name for name, digest in hashes[0].items()
                     if (REPO / name).exists()
                     and hashlib.sha256((REPO / name).read_bytes()).hexdigest() != digest)
    if drifted:
        print(f"NOTE: the W2 runs executed sources that changed later (declared in "
              f"docs/deviations.md): {', '.join(drifted)}", file=sys.stderr)
    for config in configs:
        if config["seed"] != 0 or config["variant"] != "clip0.2" or config["smoke"]:
            raise ValueError("W2 requires seed 0, clip0.2, non-smoke runs")
        for key, value in BASELINE.items():
            if config["hyperparameters"].get(key) != value:
                raise ValueError(f"baseline mismatch: {key}")
        if config["packages"].get("gymnasium") != "0.29.1" or config["packages"].get("mujoco") != "3.1.6":
            raise ValueError("baseline simulator versions differ from frozen versions")
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


def _verify_matrix(root, run_ids, epochs, tag):
    """3 environments x 3 seeds x ``epochs``, one factor changed.

    Every hyperparameter except ``update_epochs`` must be identical across cells,
    and all runs must come from the same platform, thread count and package set.
    That is what makes the epoch count the only candidate explanation for a
    difference, whether the matrix has two arms (W4/C2) or a whole sweep (W5).
    """
    epochs = tuple(epochs)
    expected = len(ENVS) * len(SEEDS) * len(epochs)
    if len(run_ids) != expected or len(set(run_ids)) != expected:
        raise ValueError(f"{tag} requires exactly {expected} distinct runs")
    configs = [verify_run(root / "runs" / rid, root) for rid in run_ids]
    cells = {(config["env"], config["seed"], config["hyperparameters"]["update_epochs"])
             for config in configs}
    wanted = {(env, seed, value) for env in ENVS for seed in SEEDS for value in epochs}
    if cells != wanted:
        raise ValueError(f"{tag} matrix incomplete: missing {sorted(wanted - cells)}, "
                         f"unexpected {sorted(cells - wanted)}")
    for config in configs:
        if config["smoke"] or config["variant"] != C2_VARIANT:
            raise ValueError(f"{tag} requires non-smoke {C2_VARIANT} runs")
        if config["hyperparameters"]["update_epochs"] not in epochs:
            raise ValueError(f"{tag} update_epochs must be one of {list(epochs)}")
        if config["steps"] != C2_STEPS:
            raise ValueError(f"{tag} requires exactly 1,000,000 steps per run")
        value = config["hyperparameters"]["update_epochs"]
        if f"_k{value}_full" not in config["run_id"]:
            raise ValueError(f"run id does not encode update_epochs={value}: {config['run_id']}")
        if config.get("label") not in (None, f"k{value}"):
            raise ValueError("recorded label disagrees with update_epochs")
    # Environment and seed are supposed to differ across cells; the arms within a
    # cell must not differ in anything but update_epochs. Output paths are excluded
    # by construction (they encode the run directory).
    reference = dict(configs[0]["hyperparameters"])
    for key in C2_NOT_A_FACTOR:
        reference.pop(key, None)
    for config in configs[1:]:
        candidate = dict(config["hyperparameters"])
        for key in C2_NOT_A_FACTOR:
            candidate.pop(key, None)
        if candidate != reference:
            differing = sorted(key for key in set(candidate) | set(reference)
                               if candidate.get(key) != reference.get(key))
            raise ValueError(f"more than update_epochs differs: {config['run_id']} "
                             f"({', '.join(differing)})")
    if len({config["threads"] for config in configs}) != 1:
        raise ValueError(f"{tag} thread counts differ")
    if any(config["packages"] != configs[0]["packages"] for config in configs):
        raise ValueError(f"{tag} package versions differ")
    if len({config["host"]["platform"] for config in configs}) != 1:
        raise ValueError(f"{tag} runs were produced on different platforms")
    return configs


def verify_c2(root=REPO, run_ids=None):
    """W4 / C2: 3 environments x 3 seeds x update_epochs {1, 10}, one factor changed.

    Both arms must come from the same machines, thread count and package set;
    every hyperparameter except ``update_epochs`` must be identical. That is what
    makes "K" the only candidate explanation for a difference.
    """
    if run_ids is None:
        manifest = json.loads((root / "logs" / "w4_c2.json").read_text(encoding="utf-8"))
        run_ids = manifest["run_ids"]
    return _verify_matrix(root, run_ids, C2_EPOCHS, "C2")


def verify_kscan(root=REPO, run_ids=None, epochs=KSCAN_EPOCHS):
    """W5 / extension: the same matrix with ``update_epochs`` swept over K values.

    The sweep shares its K = 1 and K = 10 cells with W4/C2, so those runs are
    verified here again rather than retrained; the check is the same one C2 passed.
    """
    if run_ids is None:
        manifest = json.loads((root / "logs" / "w5_extension.json").read_text(encoding="utf-8"))
        run_ids = manifest["run_ids"]
    return _verify_matrix(root, run_ids, epochs, "W5/K-scan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--c2", action="store_true")
    parser.add_argument("--kscan", action="store_true")
    parser.add_argument("--root", type=Path, default=REPO)
    args = parser.parse_args()
    if not args.run and not args.baseline and not args.c2 and not args.kscan:
        parser.error("choose --run, --baseline, --c2 or --kscan")
    try:
        if args.baseline:
            configs = verify_baseline(args.root)
        elif args.c2:
            configs = verify_c2(args.root)
        elif args.kscan:
            configs = verify_kscan(args.root)
        else:
            configs = [verify_run(args.run, args.root)]
    except (OSError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}")
        return 1
    for config in configs:
        print(f"PASS {config['run_id']}: {config['actual_steps']:,} steps, {config['updates']} updates")
    return 0


if __name__ == "__main__":
    sys.exit(main())

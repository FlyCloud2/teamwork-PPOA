#!/usr/bin/env python
"""Run the W4 / C2 matrix: update_epochs 1 vs 10, one factor changed.

Three environments x three seeds x two arms = 18 runs of exactly 1,000,000
steps. Both arms are produced by this script on the same machine and in the same
window. The K=10 arm is deliberately NOT reused from W3: those runs came from a
different host, and a host change is exactly the kind of confound a single-factor
comparison cannot afford.

Wall-clock time is recorded per run but is not compared with W3, which ran on
another machine.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
import time

from evidence import run_pattern
from run import REPO, atomic_json
from verify_runs import C2_EPOCHS, C2_STEPS, C2_VARIANT, ENVS, SEEDS, verify_c2

PACKAGES = (("gymnasium", "0.29.1"), ("mujoco", "3.1.6"),
            ("numpy", "1.26.4"), ("torch", "2.4.1"))


def matrix(root, threads):
    """Commands in an interleaved order so each wave mixes arms and environments."""
    commands = []
    for env in ENVS:
        for seed in SEEDS:
            for epochs in C2_EPOCHS:
                commands.append([
                    sys.executable, str(REPO / "scripts" / "run.py"),
                    "--env", env, "--seed", str(seed), "--variant", C2_VARIANT,
                    "--update-epochs", str(epochs), "--total-timesteps", str(C2_STEPS),
                    "--cuda", "false", "--threads", str(threads),
                    "--label", f"k{epochs}", "--output-root", str(root)])
    return commands


def existing(root):
    found = []
    for env in ENVS:
        for seed in SEEDS:
            for epochs in C2_EPOCHS:
                pattern = run_pattern(env, seed, C2_VARIANT, f"k{epochs}")
                found.extend(path.name for path in (Path(root) / "runs").glob(pattern))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=6, help="concurrent processes (1-18)")
    parser.add_argument("--threads", type=int, default=1, help="OMP/MKL threads per process")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=REPO)
    args = parser.parse_args()
    if not 1 <= args.jobs <= len(ENVS) * len(SEEDS) * len(C2_EPOCHS):
        parser.error("--jobs must be between 1 and 18")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    commands = matrix(args.output_root, args.threads)
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return 0
    for package, version in PACKAGES:
        if md.version(package).split("+")[0] != version:
            parser.error(f"{package} must be {version}")
    root = args.output_root.resolve()
    # A duplicated cell would make every summary ambiguous, so refuse to start.
    already = existing(root)
    if already:
        parser.error(f"C2 cells already exist ({len(already)} run directories, e.g. "
                     f"{already[0]}); remove them or use --output-root")
    before = {path.name for path in (root / "runs").glob("*")} if (root / "runs").exists() else set()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(lambda cmd: subprocess.call(cmd, cwd=REPO), commands))
    if any(codes):
        print(f"C2 matrix failed: exit codes {codes}; inspect runs/*/stdout.log")
        return 1
    ids = sorted(path.name for path in (root / "runs").iterdir() if path.name not in before)
    try:
        verify_c2(root, ids)
    except (OSError, ValueError, KeyError) as exc:
        print(f"C2 verification failed: {exc}")
        return 1
    atomic_json(root / "logs" / "w4_c2.json", {
        "created": dt.datetime.now().isoformat(timespec="seconds"), "run_ids": ids,
        "variant": C2_VARIANT, "epochs": list(C2_EPOCHS),
        "jobs": args.jobs, "threads_per_run": args.threads,
        "requested_steps_per_run": C2_STEPS,
        "wall_time_s": round(time.monotonic() - started, 1), "status": "verified",
    })
    print(f"C2 matrix complete and verified ({len(ids)} runs, "
          f"{(time.monotonic() - started) / 60:.1f} min); manifest: logs/w4_c2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

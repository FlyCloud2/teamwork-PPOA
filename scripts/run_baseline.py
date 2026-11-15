#!/usr/bin/env python
"""Run the W2 matrix: three MuJoCo environments, seed 0, exactly one million steps."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime as dt
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys

from run import REPO, atomic_json
from verify_runs import ENVS, verify_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=REPO)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    commands = [[sys.executable, str(REPO / "scripts" / "run.py"), "--env", env,
                 "--seed", "0", "--variant", "clip0.2", "--total-timesteps", "1000000",
                 "--cuda", "false", "--threads", str(args.threads), "--output-root", str(args.output_root)]
                for env in ENVS]
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return 0
    for package, version in (("gymnasium", "0.29.1"), ("mujoco", "3.1.6"), ("numpy", "1.26.4"), ("torch", "2.4.1")):
        if md.version(package).split("+")[0] != version:
            parser.error(f"{package} must be {version}")
    root = args.output_root.resolve()
    before = {path.name for path in (root / "runs").glob("*")} if (root / "runs").exists() else set()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(lambda cmd: subprocess.call(cmd, cwd=REPO), commands))
    if any(codes):
        print(f"Baseline failed: exit codes {codes}; inspect runs/*/stdout.log")
        return 1
    ids = sorted(path.name for path in (root / "runs").iterdir() if path.name not in before)
    try:
        verify_baseline(root, ids)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Baseline verification failed: {exc}")
        return 1
    atomic_json(root / "logs" / "w2_baseline.json", {
        "created": dt.datetime.now().isoformat(timespec="seconds"), "run_ids": ids,
        "jobs": args.jobs, "threads_per_run": args.threads,
        "requested_steps_per_run": 1_000_000, "status": "verified",
    })
    print("W2 baseline complete and verified; manifest: logs/w2_baseline.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

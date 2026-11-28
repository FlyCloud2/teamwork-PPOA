#!/usr/bin/env python
"""Run the W5 extension: sweep the epoch count K over several values.

The paper fixes K = 10 (Table 3) and never ablates it, so the extension asks a
single-factor question: is the default K a load-bearing choice, and where does it
sit on a sweep? Everything except ``update_epochs`` is held at the paper's setting
and at the W4/C2 setting — variant ``clip0.2`` (epsilon = 0.2), 3 environments x
3 seeds, exactly 1,000,000 steps per run, one OMP/MKL thread per process, one host,
the same pinned packages.

Cells that already exist on this host are reused instead of retrained: K = 1 and
K = 10 are the accepted W4/C2 runs (same machine, same budget), so the sweep only
trains the missing K values. Reuse is refused when an existing run came from
another platform, thread count or package set, because then K would no longer be
the only difference between cells.

Every run is written by ``scripts/run.py``, so each one carries its own
``config.json`` (command line, resolved hyperparameters, source hashes, seed) and
``progress.csv``; nothing here recomputes a number.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import importlib.metadata as md
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

from evidence import ENVS, SEEDS, run_pattern
from run import REPO, atomic_json
from verify_runs import (KSCAN_EPOCHS, KSCAN_STEPS, KSCAN_VARIANT, verify_kscan,
                         verify_run)

PACKAGES = (("gymnasium", "0.29.1"), ("mujoco", "3.1.6"),
            ("numpy", "1.26.4"), ("torch", "2.4.1"))


def parse_epochs(values):
    """Sorted, de-duplicated K values; at least one and all positive."""
    epochs = tuple(sorted({int(value) for value in values}))
    if not epochs or epochs[0] < 1:
        raise SystemExit("--epochs must be positive integers")
    return epochs


def label(epochs):
    """The runner label for one K; it enters run_id and never the trainer."""
    return f"k{epochs}"


def matrix(root, epochs, threads):
    """``[(cell, command)]`` for every cell, so missing cells can be filtered out."""
    cells = []
    for env in ENVS:
        for seed in SEEDS:
            for value in epochs:
                command = [
                    sys.executable, str(REPO / "scripts" / "run.py"),
                    "--env", env, "--seed", str(seed), "--variant", KSCAN_VARIANT,
                    "--update-epochs", str(value), "--total-timesteps", str(KSCAN_STEPS),
                    "--cuda", "false", "--threads", str(threads),
                    "--label", label(value), "--output-root", str(root)]
                cells.append(((env, seed, value), command))
    return cells


def discover(root, epochs):
    """``{(env, seed, K): [candidate run directories]}`` as currently on disk."""
    found = {}
    for env in ENVS:
        for seed in SEEDS:
            for value in epochs:
                pattern = run_pattern(env, seed, KSCAN_VARIANT, label(value))
                found[(env, seed, value)] = sorted(
                    path.name for path in (Path(root) / "runs").glob(pattern))
    return found


def reuse_verdict(config, threads, platform_name):
    """Why an existing run may or may not stand in for a fresh one (None = may).

    The build string is compared exactly (not normalized): a run produced with a
    different BLAS backend than the installed one is a different environment, and
    a sweep must not mix them.
    """
    if config.get("threads") != threads:
        return f"recorded {config.get('threads')} threads, wanted {threads}"
    recorded = (config.get("host") or {}).get("platform")
    if recorded != platform_name:
        return f"produced on {recorded!r}, this host is {platform_name!r}"
    for package, _ in PACKAGES:
        installed = (config.get("packages") or {}).get(package)
        current = md.version(package)
        if installed != current:
            return f"{package} {installed!r} recorded, {current!r} installed"
    return None


def manifest(ids, epochs, mode, jobs, threads, reused, launched, wall_time_s):
    """The accepted sweep, with the run list that the verifier re-checks."""
    return {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "run_ids": ids, "epochs": list(epochs), "variant": KSCAN_VARIANT,
        "steps_per_run": KSCAN_STEPS, "mode": mode, "jobs": jobs,
        "threads_per_run": threads,
        "reused": reused, "launched": launched,
        "wall_time_s": None if wall_time_s is None else round(wall_time_s, 1),
        "status": "verified",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", nargs="+", type=int, default=list(KSCAN_EPOCHS),
                        help=f"K values to sweep (default {list(KSCAN_EPOCHS)})")
    parser.add_argument("--jobs", type=int, default=6, help="concurrent processes")
    parser.add_argument("--threads", type=int, default=1, help="OMP/MKL threads per process")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-existing", action="store_true",
                        help="skip training: verify the cells on disk and rewrite the manifest")
    parser.add_argument("--output-root", type=Path, default=REPO)
    args = parser.parse_args(argv)

    epochs = parse_epochs(args.epochs)
    cells = len(ENVS) * len(SEEDS) * len(epochs)
    if not 1 <= args.jobs <= cells:
        parser.error(f"--jobs must be between 1 and {cells}")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    root = Path(args.output_root).resolve()
    platform_name = platform.platform()

    reused, blocked, missing = [], [], []
    for cell, paths in discover(root, epochs).items():
        if not paths:
            missing.append(cell)
        elif len(paths) > 1:
            blocked.append((cell, f"{len(paths)} candidate directories: {paths}"))
        else:
            try:
                config = verify_run(root / "runs" / paths[0], root)
            except (OSError, ValueError, KeyError) as exc:
                blocked.append((cell, f"{paths[0]} does not verify: {exc}"))
                continue
            reason = reuse_verdict(config, args.threads, platform_name)
            if reason:
                blocked.append((cell, f"{paths[0]}: {reason}"))
            else:
                reused.append(paths[0])

    for cell, reason in blocked:
        print(f"BLOCKED {cell[0]}/seed{cell[1]}/K={cell[2]}: {reason}")
    if blocked:
        # Mixing hosts (or silently dropping a cell) would break the one-factor
        # claim far more quietly than failing here does.
        print("Refusing to mix hosts: remove the offending run(s) or use --output-root.",
              file=sys.stderr)
        return 1

    pending = [command for cell, command in matrix(root, epochs, args.threads)
               if cell in missing]
    if args.dry_run:
        print(json.dumps({"epochs": list(epochs), "cells": cells, "reused": sorted(reused),
                          "missing": len(missing), "commands": pending}, indent=2))
        return 0

    if not args.verify_existing:
        for package, version in PACKAGES:
            if md.version(package).split("+")[0] != version:
                parser.error(f"{package} must be {version}")
    before = ({path.name for path in (root / "runs").iterdir()}
              if (root / "runs").exists() else set())
    started = time.monotonic()
    if pending and not args.verify_existing:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            codes = list(pool.map(lambda command: subprocess.call(command, cwd=REPO), pending))
        if any(codes):
            print(f"K-scan failed: exit codes {codes}; inspect runs/*/stdout.log")
            return 1
    launched = sorted(path.name for path in (root / "runs").iterdir()
                      if path.name not in before)
    elapsed = time.monotonic() - started

    ids = sorted(reused + launched)
    try:
        verify_kscan(root, ids, epochs)
    except (OSError, ValueError, KeyError) as exc:
        print(f"K-scan verification failed: {exc}")
        return 1
    mode = ("verified-existing" if args.verify_existing
            else ("launched" if launched else "reused-only"))
    atomic_json(root / "logs" / "w5_extension.json",
                manifest(ids, epochs, mode, args.jobs, args.threads, sorted(reused), launched,
                         None if args.verify_existing else elapsed))
    print(f"K-scan verified: {len(ids)} runs, {len(reused)} reused, {len(launched)} launched, "
          f"{elapsed / 60:.1f} min; manifest: logs/w5_extension.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

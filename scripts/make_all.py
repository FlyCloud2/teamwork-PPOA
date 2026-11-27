#!/usr/bin/env python
"""Regenerate every derived artifact from runs/ with one command.

Summaries, figures and tables are all script outputs; nothing under `figures/`,
`logs/` or the generated docs is hand-drawn or hand-typed. This runs the steps in
dependency order and stops at the first failure, so a stale or partial artifact
cannot be left behind unnoticed.

    python scripts/make_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

STEPS = [
    ("W2 summary", ["scripts/summarize_baseline.py"]),
    ("W2 baseline figure", ["scripts/plot.py", "--baseline", "--out", "figures/w2_baseline.png"]),
    ("C1 summary", ["scripts/summarize_c1.py"]),
    ("C1 figures", ["scripts/plot_c1.py"]),
    ("C2 summary", ["scripts/summarize_c2.py"]),
    ("C2 figures", ["scripts/plot_c2.py"]),
    ("Paper vs ours tables", ["scripts/make_table.py"]),
]

CHECKS = [
    ("W2 baseline evidence", ["scripts/verify_runs.py", "--baseline"]),
    ("W4 C2 evidence", ["scripts/verify_runs.py", "--c2"]),
]


def run(name, command):
    started = time.monotonic()
    print(f"[make_all] {name}: {' '.join(command)}", flush=True)
    code = subprocess.call([sys.executable, *command], cwd=REPO)
    elapsed = time.monotonic() - started
    if code:
        print(f"[make_all] FAILED {name} (exit {code}, {elapsed:.1f}s)", file=sys.stderr)
    return code


def main():
    for name, command in STEPS:
        if run(name, command):
            return 1
    for name, command in CHECKS:
        if run(name, command):
            return 1
    print("[make_all] PASS: summaries, figures, tables and evidence checks are current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

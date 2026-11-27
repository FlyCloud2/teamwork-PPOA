#!/usr/bin/env python
"""Verify the C1 runs and aggregate them into logs/c1_summary.json.

Every number in docs/w3_c1.md comes from this script: it reads
runs/<id>/progress.csv, re-validates each run through scripts/verify_runs.py,
and writes the summary that the report quotes. Nothing is hand-entered.

Metric: the mean raw training return over the last ten logged episodes of a
run, then mean +- std across the three seeds. Raw training returns sampled at
update boundaries are not evaluation scores.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
from verify_runs import verify_run  # noqa: E402

ENVS = ("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4")
VARIANTS = ("clip0.2", "adaptiveKL", "adaptiveKL-capped", "adaptiveKL-tight", "noclip")
SEEDS = (0, 1, 2)
# W2 baselines share the *_clip0.2_full name; C1 runs start from this date.
C1_FROM = "20261121"


def tail_metric(progress):
    values = []
    with progress.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = row["episode_return"]
            if text in ("", "nan", "None"):
                continue
            value = float(text)
            if value == value:  # not NaN
                values.append(value)
    if not values:
        raise ValueError(f"no episode returns in {progress}")
    return values[-1], statistics.fmean(values[-10:])


def main():
    rows, problems = [], []
    for env in ENVS:
        for variant in VARIANTS:
            per_seed = []
            for seed in SEEDS:
                found = sorted(p for p in REPO.joinpath("runs").glob(
                    f"*_{env}_seed{seed}_{variant}_full") if p.name[:8] >= C1_FROM)
                if len(found) != 1:
                    problems.append(f"{env}/{variant}/seed{seed}: {len(found)} runs")
                    continue
                run = found[0]
                try:
                    verify_run(run, REPO)
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{run.name}: {exc}")
                    continue
                last, tail = tail_metric(run / "progress.csv")
                per_seed.append({"seed": seed, "run": run.name,
                                 "last_return": last, "tail10_mean": tail})
            if len(per_seed) == 3:
                tails = [item["tail10_mean"] for item in per_seed]
                rows.append({"env": env, "variant": variant,
                             "mean": statistics.fmean(tails),
                             "std": statistics.stdev(tails),
                             "seeds": per_seed})

    print(f"verified groups: {len(rows)} / {len(ENVS) * len(VARIANTS)}, "
          f"problems: {len(problems)}")
    for problem in problems:
        print("  PROBLEM", problem)

    print("\n| Environment | Variant | Mean | Std | n |")
    print("|---|---|---:|---:|---:|")
    for row in rows:
        print(f"| {row['env']} | `{row['variant']}` | {row['mean']:.1f} | "
              f"{row['std']:.1f} | 3 |")

    for env in ENVS:
        block = sorted([r for r in rows if r["env"] == env],
                       key=lambda r: r["mean"], reverse=True)
        order = " > ".join(f"{r['variant']}({r['mean']:.0f})" for r in block)
        print(f"{env}: {order}")

    out = REPO / "logs" / "c1_summary.json"
    out.write_text(json.dumps({"rows": rows, "problems": problems},
                              indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"\nwritten: {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

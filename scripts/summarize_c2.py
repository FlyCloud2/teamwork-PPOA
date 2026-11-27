#!/usr/bin/env python
"""Verify the W4 / C2 runs and aggregate them into logs/c2_summary.json.

Every number in docs/w4_c2.md and docs/paper_vs_ours.md comes from this script
(or from scripts/make_table.py, which reuses the same helpers). It reads
runs/<id>/progress.csv through scripts/evidence.py, re-validates every run with
scripts/verify_runs.py, and writes the summary that the report quotes. Nothing is
hand-entered.

Metric (identical to W3/C1 so the two rounds are measured the same way): the mean
raw training return over the last ten logged episodes of a run, then mean +- std
across the three seeds. Raw training returns sampled at update boundaries are not
evaluation scores.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
from evidence import ENVS, collect, mean_diagnostics  # noqa: E402
from verify_runs import verify_run  # noqa: E402

ARMS = (1, 10)
VARIANT = "clip0.2"


def arm_payload(row):
    return {"mean": row["mean"], "std": row["std"], "wall_time_s": row["wall_time_s"],
            "seeds": row["seeds"], "diagnostics": mean_diagnostics(row)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out or (root / "logs" / "c2_summary.json")

    arms, problems = {}, []
    for epochs in ARMS:
        rows, arm_problems = collect(root, variant=VARIANT, label=f"k{epochs}",
                                     validator=verify_run)
        arms[epochs] = {row["env"]: row for row in rows}
        problems.extend(arm_problems)

    comparisons = []
    for env in ENVS:
        if env not in arms[1] or env not in arms[10]:
            continue
        one, ten = arm_payload(arms[1][env]), arm_payload(arms[10][env])
        delta = ten["mean"] - one["mean"]
        comparisons.append({
            "env": env, "k1": one, "k10": ten, "delta": delta,
            "ratio": (ten["mean"] / one["mean"]) if one["mean"] else float("nan"),
            "winner": "k10" if delta > 0 else ("k1" if delta < 0 else "tie"),
        })

    summary = {
        "metric": "mean raw training return over the last 10 logged episodes of a run, "
                  "then mean +- std over 3 seeds",
        "variant": VARIANT, "arms": list(ARMS), "steps_per_run": 1_000_000,
        "rows": comparisons, "problems": problems,
        "k10_wins": sum(1 for row in comparisons if row["winner"] == "k10"),
        "k1_wins": sum(1 for row in comparisons if row["winner"] == "k1"),
    }
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")

    print(f"verified environments: {len(comparisons)} / {len(ENVS)}, problems: {len(problems)}")
    for problem in problems:
        print("  PROBLEM", problem)
    print("\n| Environment | K=1 | K=10 | Delta (K10 - K1) | Winner |")
    print("|---|---:|---:|---:|---|")
    for row in comparisons:
        print(f"| {row['env']} | {row['k1']['mean']:.1f} +- {row['k1']['std']:.1f} | "
              f"{row['k10']['mean']:.1f} +- {row['k10']['std']:.1f} | {row['delta']:+.1f} | "
              f"{row['winner']} |")
    print("\n| Environment | clipfrac K=1 | clipfrac K=10 | approx_kl K=1 | approx_kl K=10 "
          "| wall min K=1 | wall min K=10 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in comparisons:
        one, ten = row["k1"], row["k10"]
        print(f"| {row['env']} | {one['diagnostics']['clipfrac']:.3f} | "
              f"{ten['diagnostics']['clipfrac']:.3f} | {one['diagnostics']['approx_kl']:.5f} | "
              f"{ten['diagnostics']['approx_kl']:.5f} | {one['wall_time_s'] / 60:.1f} | "
              f"{ten['wall_time_s'] / 60:.1f} |")
    print(f"\nK=10 wins in {summary['k10_wins']}/{len(comparisons)} environments, "
          f"K=1 wins in {summary['k1_wins']}/{len(comparisons)}")
    print(f"written: {out}")
    return 1 if problems or len(comparisons) != len(ENVS) else 0


if __name__ == "__main__":
    sys.exit(main())

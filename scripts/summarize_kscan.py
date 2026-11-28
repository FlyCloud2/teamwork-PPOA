#!/usr/bin/env python
"""Aggregate the W5 K sweep into logs/w5_extension_summary.json.

Every number in the extension section of the report comes from this script. It
reads ``runs/<id>/progress.csv`` through ``scripts/evidence.py``, re-validates
every run with ``scripts/verify_runs.py``, and writes the summary that the report
quotes; nothing is hand-entered.

Metric, identical to W3/W4 so the three rounds are measured the same way: the mean
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
from verify_runs import KSCAN_EPOCHS, KSCAN_STEPS, KSCAN_VARIANT, verify_run  # noqa: E402

ARMS = KSCAN_EPOCHS


def arm_payload(row):
    return {"mean": row["mean"], "std": row["std"], "wall_time_s": row["wall_time_s"],
            "seeds": row["seeds"], "diagnostics": mean_diagnostics(row)}


def rank_arms(per_env):
    """Best-to-worst K per environment, ordered by the mean over seeds."""
    return {env: [k for k, _ in sorted(arms.items(), key=lambda item: -item[1]["mean"])]
            for env, arms in per_env.items()}


def build_summary(per_env, problems, arms=ARMS):
    ranking = rank_arms(per_env)
    rows = []
    for env in ENVS:
        if env not in per_env:
            continue
        rows.append({"env": env, "arms": per_env[env],
                     "ranking": ranking[env], "best": ranking[env][0]})
    wins = {f"k{k}": sum(1 for row in rows if row["best"] == k) for k in arms}
    return {
        "metric": "mean raw training return over the last 10 logged episodes of a run, "
                  "then mean +- std over 3 seeds",
        "variant": KSCAN_VARIANT, "arms": list(arms), "steps_per_run": KSCAN_STEPS,
        "rows": rows, "problems": problems, "best_counts": wins,
        "complete": not problems and len(rows) == len(ENVS),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--arms", nargs="+", type=int, default=list(ARMS))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out or (root / "logs" / "w5_extension_summary.json")
    arms = tuple(sorted(set(args.arms)))

    per_env, problems = {}, []
    for epochs in arms:
        rows, arm_problems = collect(root, variant=KSCAN_VARIANT, label=f"k{epochs}",
                                     validator=verify_run)
        problems.extend(arm_problems)
        for row in rows:
            per_env.setdefault(row["env"], {})[epochs] = arm_payload(row)

    summary = build_summary(per_env, problems, arms)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")

    print(f"verified environments: {len(summary['rows'])} / {len(ENVS)}, "
          f"problems: {len(problems)}, arms: {list(arms)}, complete: {summary['complete']}")
    for problem in problems:
        print("  PROBLEM", problem)
    header = "| Environment | " + " | ".join(f"K={k}" for k in arms) + " | Best |"
    print("\n" + header)
    print("|---|" + "---:|" * (len(arms) + 1))
    for row in summary["rows"]:
        cells = " | ".join(f"{row['arms'][k]['mean']:.1f} +- {row['arms'][k]['std']:.1f}"
                           if k in row["arms"] else "n/a" for k in arms)
        print(f"| {row['env']} | {cells} | K={row['best']} |")
    print(f"\nbest per environment: {summary['best_counts']}")
    print(f"written: {out}")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Shared run selection and aggregation helpers for the evidence scripts.

Nothing in this module stores a result: every number is read from
runs/<run_id>/progress.csv. Selection happens on the run-directory name and is
then re-checked against config.json by the caller's validator, so a missing,
duplicated or mismatched run fails loudly instead of being silently aggregated.

Metric definition (identical to W3/C1, so tables can be compared):
the mean raw training return over the last ten logged episodes of a run, then
mean +- std across seeds. These are training returns sampled at update
boundaries; they are not evaluation scores.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENVS = ("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4")
SEEDS = (0, 1, 2)
WINDOW = 10
DIAGNOSTIC_COLUMNS = ("clipfrac", "approx_kl", "entropy", "explained_variance")


class EvidenceError(RuntimeError):
    """A required run is missing, duplicated or internally inconsistent."""


def run_pattern(env, seed, variant, label="", tag="full"):
    """Glob that a single accepted run directory must match."""
    stem = f"*_{env}_seed{seed}_{variant}" + (f"_{label}" if label else "")
    return f"{stem}_{tag}"


def find_run(root, env, seed, variant, label="", tag="full", since=""):
    """Return the one run directory for a cell; raise when it is not unique."""
    matches = sorted(path for path in (Path(root) / "runs").glob(
        run_pattern(env, seed, variant, label, tag)) if path.name[:8] >= since)
    if len(matches) != 1:
        raise EvidenceError(f"{env}/seed{seed}/{variant}"
                            f"{'/' + label if label else ''}: expected exactly one run, "
                            f"found {len(matches)}")
    return matches[0]


def episode_returns(progress):
    """All finite ``episode_return`` values of a progress.csv, in order."""
    values = []
    with Path(progress).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = row["episode_return"]
            if text in ("", "nan", "None"):
                continue
            value = float(text)
            if value == value:  # not NaN
                values.append(value)
    return values


def tail_metric(progress, window=WINDOW):
    """Mean raw training return over the last ``window`` logged episodes."""
    values = episode_returns(progress)[-window:]
    if not values:
        raise EvidenceError(f"no episode returns in {progress}")
    return statistics.fmean(values)


def diagnostics(progress, window=50):
    """Mean optimisation diagnostics over the last ``window`` updates.

    Under kl-penalty ``clipfrac`` is a pure diagnostic (the fraction of ratios
    outside the 0.2 band), not a loss term.
    """
    collected = {key: [] for key in DIAGNOSTIC_COLUMNS}
    with Path(progress).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in DIAGNOSTIC_COLUMNS:
                text = row.get(key, "")
                if text in ("", "nan", "None"):
                    continue
                value = float(text)
                if value == value:
                    collected[key].append(value)
    return {key: (statistics.fmean(values[-window:]) if values else float("nan"))
            for key, values in collected.items()}


def wall_time_s(run):
    """Recorded wall time of a finished run, in seconds (0.0 if absent)."""
    config = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
    return float(config.get("wall_time_s") or 0.0)


def collect(root, envs=ENVS, seeds=SEEDS, variant="clip0.2", label="", since="",
            validator=None):
    """Aggregate one condition over environments and seeds.

    Returns ``(rows, problems)``. A row is emitted only when every requested seed
    was found and (if ``validator`` is given) passed verification, so a partial
    matrix can never be reported as a complete one.
    """
    rows, problems = [], []
    for env in envs:
        per_seed = []
        for seed in seeds:
            try:
                run = find_run(root, env, seed, variant, label, since=since)
                if validator is not None:
                    validator(run, Path(root))
                per_seed.append({"seed": seed, "run": run.name,
                                 "wall_time_s": wall_time_s(run),
                                 "tail_mean": tail_metric(run / "progress.csv"),
                                 "diagnostics": diagnostics(run / "progress.csv")})
            except (EvidenceError, OSError, ValueError, KeyError) as exc:
                problems.append(f"{env}/{label or variant}/seed{seed}: {exc}")
        if len(per_seed) == len(seeds):
            tails = [item["tail_mean"] for item in per_seed]
            rows.append({"env": env, "condition": label or variant, "variant": variant,
                         "mean": statistics.fmean(tails),
                         "std": statistics.stdev(tails),
                         "wall_time_s": statistics.fmean(item["wall_time_s"] for item in per_seed),
                         "seeds": per_seed})
    return rows, problems


def mean_diagnostics(row):
    """Mean of the per-seed last-window diagnostics of one aggregated row."""
    return {key: statistics.fmean(item["diagnostics"][key] for item in row["seeds"])
            for key in DIAGNOSTIC_COLUMNS}


def main():
    # `python scripts/evidence.py` is a quick health check, not a report.
    rows, problems = collect(REPO, variant="clip0.2", since="20261121")
    for row in rows:
        print(f"{row['env']}: {row['mean']:.1f} +- {row['std']:.1f} (n={len(row['seeds'])})")
    for problem in problems:
        print("PROBLEM", problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

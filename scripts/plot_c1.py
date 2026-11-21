#!/usr/bin/env python
"""C1 figures: the four surrogate variants across three environments.

Reads runs/*/progress.csv only -- no hand-entered numbers. Two outputs:

  figures/c1_variants.png  grouped bars, mean +- std over seeds, of the same
                           metric the summary table uses (last ten logged
                           episode returns of a run)
  figures/c1_curves.png    learning curves, per-update episode return smoothed
                           over a rolling window, mean across seeds with a
                           +- 1 std band

Raw training returns sampled at update boundaries are NOT evaluation scores.
"""
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ENVS = ("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4")
SEEDS = (0, 1, 2)
DEFAULT_VARIANTS = ("clip0.2", "adaptiveKL", "adaptiveKL-capped",
                    "adaptiveKL-tight", "noclip")
LABELS = {"clip0.2": "clip 0.2", "adaptiveKL": "adaptive KL (literal)",
          "adaptiveKL-capped": "adaptive KL (beta<=100)",
          "adaptiveKL-tight": "adaptive KL (beta<=1)", "noclip": "no clip"}
COLORS = {"clip0.2": "#1f77b4", "adaptiveKL": "#d62728",
          "adaptiveKL-capped": "#ff7f0e", "adaptiveKL-tight": "#2ca02c",
          "noclip": "#7f7f7f"}
WINDOW = 20


def runs_for(env, variant, seed):
    matches = [p for p in REPO.joinpath("runs").glob(
        f"*_{env}_seed{seed}_{variant}_full") if p.name[:8] >= "20261121"]
    return sorted(matches)


def read_series(progress):
    steps, returns = [], []
    with progress.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            steps.append(int(row["global_step"]))
            text = row["episode_return"]
            value = float(text) if text not in ("", "nan", "None") else float("nan")
            returns.append(value)
    return steps, returns


def forward_fill(values):
    out, last = [], float("nan")
    for value in values:
        if value == value:
            last = value
        out.append(last)
    return out


def rolling(values, window):
    out = []
    for index in range(len(values)):
        chunk = [v for v in values[max(0, index - window + 1):index + 1] if v == v]
        out.append(statistics.fmean(chunk) if chunk else float("nan"))
    return out


def tail_metric(returns):
    values = [v for v in returns if v == v]
    return statistics.fmean(values[-10:]) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    args = parser.parse_args()
    variants = [v for v in args.variants]

    fig, axes = plt.subplots(1, len(ENVS), figsize=(4.6 * len(ENVS), 4.2))
    for axis, env in zip(axes, ENVS):
        width = 0.8 / max(len(variants), 1)
        for index, variant in enumerate(variants):
            seeds = []
            for seed in SEEDS:
                found = runs_for(env, variant, seed)
                if len(found) != 1:
                    continue
                _, returns = read_series(found[0] / "progress.csv")
                seeds.append(tail_metric(returns))
            if len(seeds) < 2:
                continue
            centre = statistics.fmean(seeds)
            error = statistics.stdev(seeds)
            axis.bar(index * width, centre, width * 0.9, yerr=error, capsize=4,
                     color=COLORS.get(variant, "black"), label=LABELS.get(variant, variant))
        axis.set_xticks([i * width for i in range(len(variants))])
        axis.set_xticklabels([LABELS.get(v, v) for v in variants], rotation=30,
                             ha="right", fontsize=8)
        axis.set_title(env)
        axis.set_ylabel("tail-10 training return")
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("C1: surrogate objective comparison (mean +- std over 3 seeds)")
    fig.tight_layout()
    out_bars = REPO / "figures" / "c1_variants.png"
    fig.savefig(out_bars, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ENVS), figsize=(4.6 * len(ENVS), 4.0))
    for axis, env in zip(axes, ENVS):
        for variant in variants:
            curves = []
            steps = None
            for seed in SEEDS:
                found = runs_for(env, variant, seed)
                if len(found) != 1:
                    continue
                step_axis, returns = read_series(found[0] / "progress.csv")
                steps = step_axis
                curves.append(rolling(forward_fill(returns), WINDOW))
            if not curves or steps is None:
                continue
            mean_curve = [statistics.fmean(c[i] for c in curves) for i in range(len(steps))]
            axis.plot(steps, mean_curve, label=LABELS.get(variant, variant),
                      color=COLORS.get(variant, "black"), linewidth=1.4)
            if len(curves) > 1:
                low = [min(c[i] for c in curves) for i in range(len(steps))]
                high = [max(c[i] for c in curves) for i in range(len(steps))]
                axis.fill_between(steps, low, high, alpha=0.15,
                                  color=COLORS.get(variant, "black"))
        axis.set_title(env)
        axis.set_xlabel("environment steps")
        axis.set_ylabel(f"episode return (rolling {WINDOW})")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("C1: learning curves, mean across seeds (band = min/max)")
    fig.tight_layout()
    out_curves = REPO / "figures" / "c1_curves.png"
    fig.savefig(out_curves, dpi=150)
    plt.close(fig)

    print(f"written: {out_bars}")
    print(f"written: {out_curves}")


if __name__ == "__main__":
    main()

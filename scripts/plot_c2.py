#!/usr/bin/env python
"""C2 figures: the update_epochs 1 vs 10 arms across three environments.

Reads runs/*/progress.csv only -- no hand-entered numbers. Two outputs:

  figures/<name>_epochs.png   grouped bars, mean +- std over seeds of the same
                              metric the summary table uses (last ten logged
                              episode returns), with each seed drawn as a point
  figures/<name>_curves.png   learning curves, per-update episode return smoothed
                              over a rolling window, mean across seeds with a
                              min/max band

``--arms`` selects which K values to draw and ``--name`` selects the output
prefix, so the W5 sweep (``--arms 1 3 10 20 --name w5_kscan``) can be plotted
without overwriting the accepted C2 figures.

Raw training returns sampled at update boundaries are NOT evaluation scores.
"""
import argparse
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evidence import ENVS, REPO, SEEDS, find_run  # noqa: E402

ARMS = (1, 10)
LABELS = {1: "K = 1 epoch", 10: "K = 10 epochs"}
COLORS = {1: "#ff7f0e", 10: "#1f77b4"}
WINDOW = 20


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO))
    parser.add_argument("--arms", nargs="+", type=int, default=list(ARMS))
    parser.add_argument("--name", default="c2", help="output prefix under figures/")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, len(ENVS), figsize=(4.6 * len(ENVS), 4.2))
    for axis, env in zip(axes, ENVS):
        width = 0.8 / max(len(args.arms), 1)
        for index, epochs in enumerate(args.arms):
            per_seed = []
            for seed in SEEDS:
                run = find_run(args.root, env, seed, "clip0.2", label=f"k{epochs}")
                _, returns = read_series(run / "progress.csv")
                per_seed.append(tail_metric(returns))
            centre = statistics.fmean(per_seed)
            error = statistics.stdev(per_seed)
            axis.bar(index * width, centre, width * 0.9, yerr=error, capsize=4,
                     color=COLORS.get(epochs, "black"), label=LABELS.get(epochs, f"K={epochs}"))
            axis.plot([index * width] * len(per_seed), per_seed, "o", markersize=4,
                      color="black", alpha=0.65)
        axis.set_xticks([i * width for i in range(len(args.arms))])
        axis.set_xticklabels([LABELS.get(a, f"K={a}") for a in args.arms])
        axis.set_title(env)
        axis.set_ylabel("tail-10 training return")
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{args.name.upper()}: update_epochs (K) comparison, mean +- std over 3 seeds; dots are seeds")
    fig.tight_layout()
    out_bars = REPO / "figures" / f"{args.name}_epochs.png"
    out_bars.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_bars, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ENVS), figsize=(4.6 * len(ENVS), 4.0))
    for axis, env in zip(axes, ENVS):
        for epochs in args.arms:
            curves, steps = [], None
            for seed in SEEDS:
                run = find_run(args.root, env, seed, "clip0.2", label=f"k{epochs}")
                step_axis, returns = read_series(run / "progress.csv")
                steps = step_axis
                curves.append(rolling(forward_fill(returns), WINDOW))
            if not curves or steps is None:
                continue
            mean_curve = [statistics.fmean(c[i] for c in curves) for i in range(len(steps))]
            axis.plot(steps, mean_curve, label=LABELS.get(epochs, f"K={epochs}"),
                      color=COLORS.get(epochs, "black"), linewidth=1.4)
            if len(curves) > 1:
                low = [min(c[i] for c in curves) for i in range(len(steps))]
                high = [max(c[i] for c in curves) for i in range(len(steps))]
                axis.fill_between(steps, low, high, alpha=0.15,
                                  color=COLORS.get(epochs, "black"))
        axis.set_title(env)
        axis.set_xlabel("environment steps")
        axis.set_ylabel(f"episode return (rolling {WINDOW})")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{args.name.upper()}: learning curves, mean across seeds (band = min/max)")
    fig.tight_layout()
    out_curves = REPO / "figures" / f"{args.name}_curves.png"
    fig.savefig(out_curves, dpi=150)
    plt.close(fig)

    print(f"written: {out_bars}")
    print(f"written: {out_curves}")


if __name__ == "__main__":
    main()

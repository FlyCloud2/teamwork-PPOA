#!/usr/bin/env python
"""Plot measured progress.csv data, optionally using the verified W2 manifest."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from run import REPO
from verify_runs import verify_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10, help="trailing update window, not episodes")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")
    if bool(args.run) == args.baseline:
        parser.error("choose exactly one of --run and --baseline")
    if args.baseline:
        configs = sorted(verify_baseline(args.root), key=lambda config: config["env"])
        paths = [args.root / "runs" / config["run_id"] for config in configs]
    else:
        paths = [args.run]
    fig, axes = plt.subplots(1, len(paths), figsize=(5.3 * len(paths), 4.0), squeeze=False)
    for axis, path in zip(axes[0], paths):
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        data = pd.read_csv(path / "progress.csv")
        axis.plot(data.global_step, data.episode_return, color="#64748b", alpha=0.35, linewidth=0.8, label="Latest completed episode")
        axis.plot(data.global_step, data.episode_return.rolling(args.window, min_periods=1).mean(),
                  color="#2563eb", linewidth=1.6, label=f"Trailing {args.window} updates")
        axis.set_title(f"{config['env']} | seed {config['seed']}")
        axis.set_xlabel("Environment steps")
        axis.set_ylabel("Raw training episode return")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle("PPO training baseline | single seed\nNo evaluation or multi-seed aggregation", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

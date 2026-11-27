#!/usr/bin/env python
"""Generate the Paper-vs-Ours tables from runs/*/progress.csv.

Writes `docs/paper_vs_ours.md` (generated; never hand-edited) and
`logs/paper_vs_ours.json`. Paper numbers come from `configs/paper_table1.json`,
which records the exact source (PPO paper, Table 1) and the team's cross-check.

What is compared, and what is not. The paper's number is an average normalised
score over 7 MuJoCo tasks, normalised with a random-policy zero point and an
unpublished best-result scale. Ours are raw training returns on 3 tasks, also
mean +- std over 3 seeds. Absolute values are therefore NOT comparable; the claim
under test is an ordering, which Table 3 checks per task mechanically.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
from evidence import ENVS, collect, mean_diagnostics  # noqa: E402
from verify_runs import verify_run  # noqa: E402

PAPER = REPO / "configs" / "paper_table1.json"
C1_FROM = "20261121"          # W3/C1 runs; the W2 seed-0 baselines share the clip0.2 name
C1_TIERS = ("clip0.2", "adaptiveKL", "noclip")
C2_ARMS = (1, 10)
C2_VARIANT = "clip0.2"


def fmt(mean, std, digits=1):
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def load_paper(path=PAPER):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if {cell["condition"] for cell in data["cells"]} != set(C1_TIERS):
        raise SystemExit("configs/paper_table1.json must describe exactly "
                         "clip0.2 / adaptiveKL / noclip")
    return data


def gather_c1(root):
    tiers, problems = {}, []
    for tier in C1_TIERS:
        rows, found = collect(root, variant=tier, since=C1_FROM, validator=verify_run)
        tiers[tier] = {row["env"]: row for row in rows}
        problems.extend(found)
    return tiers, problems


def gather_c2(root):
    arms, problems = {}, []
    for epochs in C2_ARMS:
        rows, found = collect(root, variant=C2_VARIANT, label=f"k{epochs}", validator=verify_run)
        arms[epochs] = {row["env"]: row for row in rows}
        problems.extend(found)
    return arms, problems


def ranks(tiers):
    out = {}
    for env in ENVS:
        means = {tier: tiers[tier][env]["mean"] for tier in C1_TIERS}
        order = sorted(means, key=means.get, reverse=True)
        out[env] = {tier: order.index(tier) + 1 for tier in C1_TIERS}
    return out


def normalised(tiers):
    """Best tier per task = 1.00, then averaged over tasks.

    A ratio, not the paper's shift-and-scale score: we never evaluated an
    untrained policy, so there is no random-policy zero point.
    """
    per_task = {}
    for env in ENVS:
        best = max(tiers[tier][env]["mean"] for tier in C1_TIERS)
        if best <= 0:
            raise SystemExit(f"cannot normalise {env}: the best mean is not positive")
        per_task[env] = {tier: tiers[tier][env]["mean"] / best for tier in C1_TIERS}
    return ({tier: sum(per_task[env][tier] for env in ENVS) / len(ENVS) for tier in C1_TIERS},
            per_task)


def subclaims(tiers):
    pairs = [("clip0.2", "noclip", "clipping (eps = 0.2) beats no clipping"),
             ("clip0.2", "adaptiveKL", "clipping (eps = 0.2) beats adaptive KL"),
             ("adaptiveKL", "noclip", "adaptive KL beats no clipping")]
    rows = []
    for better, worse, text in pairs:
        held = [env for env in ENVS if tiers[better][env]["mean"] > tiers[worse][env]["mean"]]
        status = ("consistent" if len(held) == len(ENVS)
                  else ("inconsistent" if not held else "partial"))
        rows.append({"claim": text, "held_in": len(held), "of": len(ENVS),
                     "environments": held, "status": status})
    held = [env for env in ENVS
            if tiers["clip0.2"][env]["mean"] > tiers["adaptiveKL"][env]["mean"]
            > tiers["noclip"][env]["mean"]]
    status = ("consistent" if len(held) == len(ENVS)
              else ("inconsistent" if not held else "partial"))
    rows.append({"claim": "the full ordering clip > adaptive KL > no clipping",
                 "held_in": len(held), "of": len(ENVS), "environments": held, "status": status})
    return rows


def c2_rows(arms):
    rows = []
    for env in ENVS:
        one, ten = arms[1][env], arms[10][env]
        delta = ten["mean"] - one["mean"]
        rows.append({"env": env, "k1": one, "k10": ten, "delta": delta,
                     "winner": "k10" if delta > 0 else ("k1" if delta < 0 else "tie")})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out_md = args.out or (root / "docs" / "paper_vs_ours.md")
    out_json = args.json or (root / "logs" / "paper_vs_ours.json")

    paper = load_paper()
    tiers, c1_problems = gather_c1(root)
    arms, c2_problems = gather_c2(root)
    problems = c1_problems + c2_problems
    if problems:
        print(f"{len(problems)} run problem(s); no table is written:", file=sys.stderr)
        for problem in problems:
            print("  PROBLEM", problem, file=sys.stderr)
        return 1

    rank = ranks(tiers)
    norm, norm_per_task = normalised(tiers)
    claims = subclaims(tiers)
    c2 = c2_rows(arms)
    k10_wins = sum(1 for row in c2 if row["winner"] == "k10")

    paper_cells = {cell["condition"]: cell for cell in paper["cells"]}
    lines = []
    add = lines.append
    add("# Paper vs ours (generated)")
    add("")
    add("> **Generated file.** Produced by `scripts/make_table.py` from "
        "`runs/*/progress.csv` and `configs/paper_table1.json`.")
    add("> Do not edit by hand; re-run the script instead:")
    add(">")
    add("> ```powershell")
    add("> python scripts/make_table.py")
    add("> ```")
    add("")
    add(f"Paper numbers: {paper['source']['paper']}, {paper['source']['location']} "
        f"([arXiv]({paper['source']['url']})).")
    add(f"Source of the metric: {paper['paper_metric']}. "
        f"Paper normalisation: {paper['paper_normalisation']}.")
    add("")

    add("## What is and is not comparable")
    add("")
    add("| | Paper (Table 1) | This replication |")
    add("|---|---|---|")
    add("| Tasks | 7 MuJoCo `-v1` tasks | 3 tasks: HalfCheetah-v4, Hopper-v4, Walker2d-v4 |")
    add("| Steps per run | 1,000,000 | 1,000,000 exact (489 updates, 576-step final batch) |")
    add("| Seeds | 3 | 3 (0 / 1 / 2) |")
    add("| Score | last 100 episodes, per-task shift-and-scale (random = 0, best = 1), "
        "averaged over 21 runs | raw training return, last 10 logged episodes, "
        "mean ± std over seeds |")
    add("| Reported variance | none | mean ± std over 3 seeds |")
    add("| Environment versions | MuJoCo 1.x / `-v1` | Gymnasium 0.29.1, MuJoCo 3.1.6, `-v4` |")
    add("")
    add("Absolute values are **not** comparable across the two columns: the paper's score "
        "is normalised against a random policy and an unpublished best result, and our "
        "environments, wrappers and simulator are a decade newer. What is comparable is the "
        "**ordering of the three objectives within each setting**. That is the claim tested "
        "below; see `docs/deviations.md` for the full list of differences.")
    add("")

    add("## Table 1. Paper Table 1 vs ours (C1: three surrogate objectives)")
    add("")
    add("| Objective | Paper (7 tasks, normalised) | Ours (3 tasks, ratio in this experiment) "
        "| Ours rank per task (HalfCheetah / Hopper / Walker2d) |")
    add("|---|---:|---:|---|")
    for tier in C1_TIERS:
        cell = paper_cells[tier]
        ranks_text = " / ".join(str(rank[env][tier]) for env in ENVS)
        add(f"| {cell['paper_label']} | {cell['paper_score']:+.2f} | {norm[tier]:.2f} "
            f"| {ranks_text} |")
    add("")
    add("The `ratio` column divides each objective's per-task mean return by the best "
        "objective's mean return in the same task and averages over the 3 tasks. It is not "
        "the paper's shift-and-scale score: we never evaluated an untrained policy, so there "
        "is no random-policy zero point. A ratio below 1 means \"worse than the best "
        "objective on this task\", not \"worse than random\".")
    add("")

    add("## Table 2. Per-task C1 detail (raw training return, mean ± std over 3 seeds)")
    add("")
    add("| Task | " + " | ".join(paper_cells[tier]["paper_label"] for tier in C1_TIERS)
        + " | Best |")
    add("|---|" + "---:|" * (len(C1_TIERS) + 1))
    for env in ENVS:
        cells = " | ".join(fmt(tiers[tier][env]["mean"], tiers[tier][env]["std"])
                           for tier in C1_TIERS)
        best = paper_cells[min(C1_TIERS, key=lambda tier: rank[env][tier])]["paper_label"]
        add(f"| {env} | {cells} | {best} |")
    add("")
    add("The three beta settings `adaptiveKL`, `adaptiveKL-capped` and `adaptiveKL-tight` "
        "produce identical trajectories in this implementation, so the adaptive-KL tier is "
        "inconclusive rather than negative: see `docs/w3_c1.md` §3 for the row-wise diff "
        "and the gradient-clipping explanation. The numbers for the two controlled variants "
        "are in `logs/c1_summary.json`.")
    add("")
    add("## Table 3. Ordering checks computed from Table 2")
    add("")
    add("| Sub-claim | Holds in | Status (data only) |")
    add("|---|---:|---|")
    for claim in claims:
        add(f"| {claim['claim']} | {claim['held_in']}/{claim['of']} tasks | {claim['status']} |")
    add("")
    add("`Status` is mechanical: `consistent` when the paper's ordering holds in all three "
        "tasks, `partial` when it holds in some, `inconsistent` when it holds in none. The "
        "interpretive verdict —including why one tier is `inconclusive` —is in "
        "`docs/w3_c1.md` §4 and the report's results section.")
    add("")

    add("## Table 4. C2: what does repeating the batch for K epochs buy?")
    add("")
    add(f"Both arms were produced by `scripts/run_c2.py` on one host, {C2_VARIANT} everywhere, "
        "3 environments × 3 seeds × 1,000,000 steps; only `update_epochs` differs. The K=10 "
        "arm is not reused from W3 because those runs came from another machine.")
    add("")
    add("### Table 4a. Returns")
    add("")
    add("| Task | K = 1 | K = 10 | Delta (K10 - K1) | Winner |")
    add("|---|---:|---:|---:|---|")
    for row in c2:
        add(f"| {row['env']} | {fmt(row['k1']['mean'], row['k1']['std'])} "
            f"| {fmt(row['k10']['mean'], row['k10']['std'])} | {row['delta']:+.1f} "
            f"| {row['winner']} |")
    add("")
    add(f"K = 10 wins in {k10_wins}/{len(c2)} tasks, K = 1 wins in "
        f"{sum(1 for row in c2 if row['winner'] == 'k1')}/{len(c2)}.")
    add("")
    add("### Table 4b. Diagnostics and cost (diagnostics = mean over the last 50 updates)")
    add("")
    add("| Task | clipfrac K=1 | clipfrac K=10 | approx_kl K=1 | approx_kl K=10 "
        "| wall min K=1 | wall min K=10 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for row in c2:
        one, ten = row["k1"], row["k10"]
        d1, d10 = mean_diagnostics(one), mean_diagnostics(ten)
        add(f"| {row['env']} | {d1['clipfrac']:.3f} | {d10['clipfrac']:.3f} "
            f"| {d1['approx_kl']:.5f} | {d10['approx_kl']:.5f} "
            f"| {one['wall_time_s'] / 60:.1f} | {ten['wall_time_s'] / 60:.1f} |")
    add("")
    add("Wall time is wall-clock on one host with one OMP/MKL thread per process and "
        "several processes running concurrently, so it is a relative cost indicator inside "
        "this matrix, not a benchmark.")
    add("")

    add("## Table 5. Provenance: the runs behind every cell")
    add("")
    add("| Condition | Run directories |")
    add("|---|---|")
    for tier in C1_TIERS:
        ids = ", ".join(f"`{tiers[tier][env]['seeds'][index]['run']}`"
                        for env in ENVS for index in range(len(tiers[tier][env]["seeds"])))
        add(f"| C1 `{tier}` | {ids} |")
    for epochs in C2_ARMS:
        ids = ", ".join(f"`{arms[epochs][env]['seeds'][index]['run']}`"
                        for env in ENVS for index in range(len(arms[epochs][env]["seeds"])))
        add(f"| C2 `K={epochs}` | {ids} |")
    add("")
    add("## Regenerate every derived artifact")
    add("")
    add("```powershell")
    add("python scripts/make_all.py     # summaries, figures and this table from runs/")
    add("```")
    add("")
    add(f"Machine-readable form: [`logs/paper_vs_ours.json`](../logs/paper_vs_ours.json). "
        f"Run index: [`logs/runs.csv`](../logs/runs.csv).")
    add("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    payload = {
        "generated_by": "scripts/make_table.py",
        "paper": paper,
        "c1": {"metric": "tail-10 raw training return, mean +- std over 3 seeds",
               "rows": [{"env": env,
                         "tiers": {tier: {"mean": tiers[tier][env]["mean"],
                                          "std": tiers[tier][env]["std"],
                                          "rank": rank[env][tier],
                                          "ratio": norm_per_task[env][tier],
                                          "runs": [item["run"] for item in tiers[tier][env]["seeds"]]}
                                   for tier in C1_TIERS}} for env in ENVS],
               "normalised_mean": norm,
               "subclaims": claims},
        "c2": {"metric": "tail-10 raw training return, mean +- std over 3 seeds",
               "rows": [{"env": row["env"],
                         "k1": {"mean": row["k1"]["mean"], "std": row["k1"]["std"],
                                "wall_time_s": row["k1"]["wall_time_s"],
                                "diagnostics": mean_diagnostics(row["k1"]),
                                "runs": [item["run"] for item in row["k1"]["seeds"]]},
                         "k10": {"mean": row["k10"]["mean"], "std": row["k10"]["std"],
                                 "wall_time_s": row["k10"]["wall_time_s"],
                                 "diagnostics": mean_diagnostics(row["k10"]),
                                 "runs": [item["run"] for item in row["k10"]["seeds"]]},
                         "delta": row["delta"], "winner": row["winner"]} for row in c2],
               "k10_wins": k10_wins},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")

    print("| Objective | Paper | Ours (ratio) | Rank HC/Ho/Wa |")
    for tier in C1_TIERS:
        ranks_text = " / ".join(str(rank[env][tier]) for env in ENVS)
        print(f"| {tier} | {paper_cells[tier]['paper_score']:+.2f} | {norm[tier]:.2f} | {ranks_text} |")
    print(f"\nC2: K=10 wins {k10_wins}/{len(c2)} tasks")
    print(f"written: {out_md}")
    print(f"written: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

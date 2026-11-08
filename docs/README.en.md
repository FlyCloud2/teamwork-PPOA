# MIAE01 — Research Paper Replication: PPO on MuJoCo

Course project (teamwork-PPOA). We reproduce **Proximal Policy Optimization**
(arXiv:1707.06347v2) from the paper itself — not by calling a framework's `PPO()` —
and we ask whether the paper's central claim about *which surrogate objective works*
survives our own measurements.

> **Status: week 1 of 7 (11/02 - 11/08), in progress.** The repository skeleton, the vendored trainer
> and the run/logging pipeline exist; the experiment matrix has not started. Nothing in
> this README is a result — see [Roadmap](#roadmap) for what is due when.

## What the paper claims, and what we test

PPO's contribution is a family of surrogate objectives that keep a policy update inside
a trust region without solving a constrained optimisation problem. The paper compares
three of them on MuJoCo (Table 1): the **clipped objective** with epsilon = 0.2, an
**adaptive KL penalty** with a target KL, and **no clipping** at all. Its claim is an
*ordering*, not a number: clipping > KL penalty > no clipping.

That ordering is what we try to falsify, in three stages:

| # | Question | Configuration |
|---|---|---|
| C1 | Does the ordering hold? | `clip0.2` vs `adaptiveKL` vs `noclip`, 3 envs x 3 seeds |
| C2 | Does re-using each batch for several epochs actually buy anything? | `update_epochs = 1` vs `10` |
| C3 | Is the paper's untuned knob (`K`, epochs) load-bearing? | `K` scan, since the paper never ablates it |

The paper's own mechanism is the baseline of every comparison, so a failure to reproduce
shows up as a difference in *ordering*, not as "our number is lower".

## Scope: what we deliberately do not attempt

* **3 of the paper's 7 MuJoCo environments** (HalfCheetah, Hopper, Walker2d). Reduced
  scale, stated as such — the report never compares our cells with the paper's cells.
* **Absolute scores are not comparable** with the paper: the wrappers, the MuJoCo
  version and the framework all differ, and each difference is recorded in
  [`docs/deviations.md`](docs/deviations.md) before the experiment that needs it.
* **No bit-for-bit reproduction** is claimed. Three seeds per configuration; we report
  mean +/- std and interpret the ordering, not the digits.

## Roadmap

One week, one deliverable. Every checkpoint is a hard deadline (Sunday 23:59), and each
member pushes their own substantive commit before it.

| Week | Dates (check) | Goal | Deliverable |
|---|---|---|---|
| **W1** | 11/02 – 11/08 | Repository foundation, environment pinned, pipeline proven | `env.txt`, one smoke run's `progress.csv` + `config.json` + `stdout.log`, first curve, deviations table started |
| W2 | 11/09 – 11/15 | Baseline at full scale | 3 envs x seed 0 x 1M steps; `logs/runs.csv` index; every knob a CLI flag; gymnasium / MuJoCo versions frozen |
| W3 | 11/16 – 11/22 | **C1: the paper's main claim** | `clip0.2` / `adaptiveKL` / `noclip` x 3 envs x 3 seeds = 27 runs, each variant a flag, no source edits |
| W4 | 11/23 – 11/29 | **C2** + automated tables | `update_epochs` 1 vs 10; `scripts/make_table.py` emits the paper-vs-ours table; report writing starts |
| W5 | 11/30 – 12/06 | Extension (single factor) | `K` scan, 3 envs x 3 seeds, one figure, one sentence of conclusion |
| W6 | 12/07 – 12/13 | Attribution + draft | At least one controlled re-run as evidence for a deviation; deviations finalised; full draft |
| W7 | 12/14 – 12/20 | Final | Report PDF, presentation, `v1.0` tag, an outsider walks this README end to end |

> The per-week execution checklist (what is due this week, weekly routine, milestones) is in `ROADMAP.md` (Chinese).

> Dates assume the whole plan slipped by one week, so week 1 starts 11/02; each week still runs Monday 00:00 to Sunday 23:59. If the presentation date stays fixed at 12/12, weeks 6 and 7 have to be merged into 12/07 - 12/12.

## Quick start

Python 3.10, versions pinned with `==` in [`requirements.txt`](requirements.txt); the
machine and every installed distribution are in [`env.txt`](env.txt).

```powershell
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

python scripts/run.py --env HalfCheetah-v4 --seed 0 --variant clip0.2 --smoke
python scripts/plot.py --run runs/<run_id> --out figures/w1_smoke_HalfCheetah.png
```

**Expected output and duration**

| Command | Steps | Expected wall time | Output |
|---|---|---|---|
| `run.py --smoke` | 20,000 (9 updates) | about a minute | `runs/<id>/progress.csv` with 9 rows, `config.json`, `stdout.log` |
| `run.py` (full) | 1,000,000 | ~55 min on the reference machine | ~488 updates, one row each |

Each run is self-describing; that is the point of the repository:

```
runs/20261030-1247_HalfCheetah-v4_seed0_clip0.2_smoke/
├─ config.json     argv, resolved hyperparameters, git commit, host, wall time, rc
├─ progress.csv    one row per update: return, losses, approx_kl, clipfrac, ...
├─ stdout.log      raw console output of the child process
└─ tensorboard/    event files, gitignored (binary, unreadable in a diff)
```

## Repository layout

```
src/ppo_continuous_action.py   vendored CleanRL PPO + [REPRO] logging hooks
scripts/run.py                 the only entry point that runs training
scripts/plot.py                progress.csv -> PNG (figures are never hand-drawn)
scripts/make_table.py          runs -> paper-vs-ours table
scripts/collect_env.py         machine + library snapshot -> env.txt
scripts/weekly_check.ps1       pre-push audit mapped to the grading rules
configs/base.yaml              paper Table 3 + experiment matrix, as recorded
docs/setup.md                  what actually happened while installing
docs/deviations.md             every difference from the paper
runs/                          committed evidence, one directory per run
logs/runs.csv                  index of every run
```

## Rules this repository enforces on itself

1. One command in, one self-describing run directory out — no manually assembled logs.
2. No hard-coded paths and no hard-coded hyperparameters; the runner passes and records
   them, and `config.json` is the single source of truth for what actually ran.
3. Every figure is generated by a script from a committed CSV.
4. Every number in the report traces back to a committed `progress.csv`.
5. Every deviation from the paper is written down *before* the experiment, in
   [`docs/deviations.md`](docs/deviations.md), not explained away afterwards.

## Upstream code and licence

`src/ppo_continuous_action.py` is CleanRL's `ppo_continuous_action.py` at commit
`cf20043b852deedf52dc9f90a325c6866d599005`, vendored rather than installed so the
algorithm cannot move under our results. All local modifications are marked `[REPRO]` in
the source and itemised in [`MODIFICATIONS.md`](MODIFICATIONS.md). Upstream licence (MIT)
in [`LICENSES/cleanrl-LICENSE.txt`](LICENSES/cleanrl-LICENSE.txt); our code is MIT, see
[`LICENSE`](LICENSE).

## Team

| Student ID | Name | Focus |
|---|---|---|
| 待填 | 待填 | 待填 |
| 待填 | 待填 | 待填 |
| 待填 | 待填 | 待填 |

## AI usage

Code and prose here are drafted with the help of an AI assistant; no experimental result
is produced by it. The per-file record, including what each member reviewed and ran, is
in [`AI_USAGE.md`](AI_USAGE.md).

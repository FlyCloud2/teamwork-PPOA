# MIAE01 — Research Paper Replication: PPO on MuJoCo

Course project (teamwork-PPOA). We reproduce **Proximal Policy Optimization**
(arXiv:1707.06347v2) from the paper itself — not by calling a framework's `PPO()` —
and we ask whether the paper's central claim about *which surrogate objective works*
survives our own measurements.

> **Status: W3 (C1) complete - partially reproduced.** 27 runs of
> `clip0.2` / `adaptiveKL` / `noclip` across 3 environments x 3 seeds x 1,000,000
> steps, plus 18 controlled runs with a bounded beta. Verdict: clipping beats
> both alternatives, but the middle position could not be tested - all three beta
> settings produce bit-identical trajectories. See the
> [generated evidence report](w3_c1.md) and the [W2 baseline](w2_baseline.md).

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
  [deviations](deviations.md) before the experiment that needs it.
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

Use Python 3.10. On Windows, put the virtual environment in an ASCII-only path;
MuJoCo 3.1.6 failed to load XML assets from the Chinese environment path on the
actual test machine. Follow [setup notes](setup.md). Runtime versions are in
[requirements.txt](../requirements.txt); the full tested dependency set is in
[requirements-lock.txt](../requirements-lock.txt) and [env.txt](../env.txt).

```powershell
python -m pip install -r requirements-lock.txt
python scripts/run.py --env HalfCheetah-v4 --seed 0 --variant clip0.2 --smoke --threads 1
python scripts/plot.py --run runs/<run_id> --out figures/smoke.png
python -m unittest discover -s tests -v
python scripts/run_baseline.py --jobs 3 --threads 1
python scripts/verify_runs.py --baseline
python scripts/plot.py --baseline --out figures/w2_baseline.png
python scripts/summarize_baseline.py
```

Use `--jobs 1` for sequential execution. The matrix entry point defaults to one
thread per process; the single-run entry point retains its previous four-thread
default. CPU is the default device; `--cuda true` explicitly requests CUDA and
fails when unavailable. No CUDA performance comparison is claimed.

The smoke run consumes exactly 20,000 steps (10 updates); the full run consumes
exactly 1,000,000 steps (489 updates). The last full-run rollout has 576 samples
and 32 minibatches of 18 samples. This differs from the vendored floor-budget
loop; see [deviations](deviations.md). The measured smoke took approximately
91 seconds including startup; full-run durations are recorded in their configs.

`python scripts/run.py --help` lists all exposed training flags, including epoch
count, rollout shape, optimizer, GAE, clipping, network widths/activation and
boolean toggles. Runner and trainer share one parser, so `config.json` records
actual child arguments instead of copied constants. `--dry-run` writes no run
artifacts. `--output-root` isolates test output.

Each run directory contains `config.json`, `progress.csv`, `stdout.log` and an
ignored `tensorboard/` directory. The config includes source SHA-256 hashes,
package versions, requested/actual counts and running/completed/failed status.
The locked index `logs/runs.csv` supports concurrent runs. The acceptance manifest
`logs/w2_baseline.json` includes only the three verified full baselines; failed
and smoke attempts remain indexed separately.

**Variant boundary:** W2 accepted only `clip0.2`; its legacy `adaptiveKL` was KL
early stopping and is now called `kl-early-stop`. W3 adds a real KL penalty as
`adaptiveKL` (literal Algorithm 2, no beta bound), plus two controls with the
bound at `1/d_target = 100` and at `1`. The three settings turned out to be
numerically identical (see [w3_c1.md](w3_c1.md) §3), so this variant is not a
valid test of the paper's middle position. Use `--variant custom` for a
non-default clipping coefficient.

Raw training returns sampled at update boundaries are not evaluation scores.
Single-seed results do not establish the C1 ordering. AI assistance and the
pending human sign-off are recorded in [AI_USAGE.md](../AI_USAGE.md).

# Paper vs ours (generated)

> **Generated file.** Produced by `scripts/make_table.py` from `runs/*/progress.csv` and `configs/paper_table1.json`.
> Do not edit by hand; re-run the script instead:
>
> ```powershell
> python scripts/make_table.py
> ```

Paper numbers: Schulman, Wolski, Dhariwal, Radford, Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347v2, Table 1, the row "PPO on MuJoCo" ([arXiv](https://arxiv.org/abs/1707.06347)).
Source of the metric: average normalised score over 7 MuJoCo tasks (-v1), 1,000,000 steps per run, 3 seeds, scored on the last 100 episodes; the paper reports one aggregate number per setting, without variance. Paper normalisation: per-task shift-and-scale with a random policy = 0 and the best result = 1, then averaged over 21 runs; the reference scores behind that scaling are not published.

## What is and is not comparable

| | Paper (Table 1) | This replication |
|---|---|---|
| Tasks | 7 MuJoCo `-v1` tasks | 3 tasks: HalfCheetah-v4, Hopper-v4, Walker2d-v4 |
| Steps per run | 1,000,000 | 1,000,000 exact (489 updates, 576-step final batch) |
| Seeds | 3 | 3 (0 / 1 / 2) |
| Score | last 100 episodes, per-task shift-and-scale (random = 0, best = 1), averaged over 21 runs | raw training return, last 10 logged episodes, mean ± std over seeds |
| Reported variance | none | mean ± std over 3 seeds |
| Environment versions | MuJoCo 1.x / `-v1` | Gymnasium 0.29.1, MuJoCo 3.1.6, `-v4` |

Absolute values are **not** comparable across the two columns: the paper's score is normalised against a random policy and an unpublished best result, and our environments, wrappers and simulator are a decade newer. What is comparable is the **ordering of the three objectives within each setting**. That is the claim tested below; see `docs/deviations.md` for the full list of differences.

## Table 1. Paper Table 1 vs ours (C1: three surrogate objectives)

| Objective | Paper (7 tasks, normalised) | Ours (3 tasks, ratio in this experiment) | Ours rank per task (HalfCheetah / Hopper / Walker2d) |
|---|---:|---:|---|
| Clipping, eps = 0.2 | +0.82 | 1.00 | 1 / 1 / 1 |
| Adaptive KL, d_targ = 0.01 | +0.74 | 0.24 | 2 / 2 / 3 |
| No clipping or penalty | -0.39 | 0.19 | 3 / 3 / 2 |

The `ratio` column divides each objective's per-task mean return by the best objective's mean return in the same task and averages over the 3 tasks. It is not the paper's shift-and-scale score: we never evaluated an untrained policy, so there is no random-policy zero point. A ratio below 1 means "worse than the best objective on this task", not "worse than random".

## Table 2. Per-task C1 detail (raw training return, mean ± std over 3 seeds)

| Task | Clipping, eps = 0.2 | Adaptive KL, d_targ = 0.01 | No clipping or penalty | Best |
|---|---:|---:|---:|---:|
| HalfCheetah-v4 | 1945.0 ± 1006.1 | 596.2 ± 109.8 | 375.0 ± 422.4 | Clipping, eps = 0.2 |
| Hopper-v4 | 1430.7 ± 726.3 | 479.6 ± 327.4 | 355.4 ± 451.0 | Clipping, eps = 0.2 |
| Walker2d-v4 | 2297.5 ± 718.5 | 209.8 ± 181.2 | 261.9 ± 227.0 | Clipping, eps = 0.2 |

The three beta settings `adaptiveKL`, `adaptiveKL-capped` and `adaptiveKL-tight` produce identical trajectories in this implementation, so the adaptive-KL tier is inconclusive rather than negative: see `docs/w3_c1.md` §3 for the row-wise diff and the gradient-clipping explanation. The numbers for the two controlled variants are in `logs/c1_summary.json`.

## Table 3. Ordering checks computed from Table 2

| Sub-claim | Holds in | Status (data only) |
|---|---:|---|
| clipping (eps = 0.2) beats no clipping | 3/3 tasks | consistent |
| clipping (eps = 0.2) beats adaptive KL | 3/3 tasks | consistent |
| adaptive KL beats no clipping | 2/3 tasks | partial |
| the full ordering clip > adaptive KL > no clipping | 2/3 tasks | partial |

`Status` is mechanical: `consistent` when the paper's ordering holds in all three tasks, `partial` when it holds in some, `inconsistent` when it holds in none. The interpretive verdict —including why one tier is `inconclusive` —is in `docs/w3_c1.md` §4 and the report's results section.

## Table 4. C2: what does repeating the batch for K epochs buy?

Both arms were produced by `scripts/run_c2.py` on one host, clip0.2 everywhere, 3 environments × 3 seeds × 1,000,000 steps; only `update_epochs` differs. The K=10 arm is not reused from W3 because those runs came from another machine.

### Table 4a. Returns

| Task | K = 1 | K = 10 | Delta (K10 - K1) | Winner |
|---|---:|---:|---:|---|
| HalfCheetah-v4 | 772.7 ± 32.2 | 2247.8 ± 1411.8 | +1475.2 | k10 |
| Hopper-v4 | 1282.8 ± 156.8 | 1775.4 ± 389.7 | +492.5 | k10 |
| Walker2d-v4 | 1046.2 ± 436.0 | 3688.0 ± 856.6 | +2641.8 | k10 |

K = 10 wins in 3/3 tasks, K = 1 wins in 0/3.

### Table 4b. Diagnostics and cost (diagnostics = mean over the last 50 updates)

| Task | clipfrac K=1 | clipfrac K=10 | approx_kl K=1 | approx_kl K=10 | wall min K=1 | wall min K=10 |
|---|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v4 | 0.000 | 0.056 | 0.00007 | 0.00725 | 20.0 | 29.5 |
| Hopper-v4 | 0.000 | 0.030 | 0.00004 | 0.00477 | 22.0 | 30.9 |
| Walker2d-v4 | 0.000 | 0.046 | 0.00006 | 0.00679 | 21.8 | 28.3 |

Wall time is wall-clock on one host with one OMP/MKL thread per process and several processes running concurrently, so it is a relative cost indicator inside this matrix, not a benchmark.

## Table 5. Provenance: the runs behind every cell

| Condition | Run directories |
|---|---|
| C1 `clip0.2` | `20261121-215526-109163_HalfCheetah-v4_seed0_clip0.2_full`, `20261121-222753-880489_HalfCheetah-v4_seed1_clip0.2_full`, `20261121-230012-819078_HalfCheetah-v4_seed2_clip0.2_full`, `20261121-215526-155533_Hopper-v4_seed0_clip0.2_full`, `20261121-223018-319976_Hopper-v4_seed1_clip0.2_full`, `20261121-230452-762856_Hopper-v4_seed2_clip0.2_full`, `20261121-215526-141378_Walker2d-v4_seed0_clip0.2_full`, `20261121-223024-522085_Walker2d-v4_seed1_clip0.2_full`, `20261121-230510-157697_Walker2d-v4_seed2_clip0.2_full` |
| C1 `adaptiveKL` | `20261121-215526-114994_HalfCheetah-v4_seed0_adaptiveKL_full`, `20261121-222757-404490_HalfCheetah-v4_seed1_adaptiveKL_full`, `20261121-230031-179462_HalfCheetah-v4_seed2_adaptiveKL_full`, `20261121-215526-155533_Hopper-v4_seed0_adaptiveKL_full`, `20261121-223018-382754_Hopper-v4_seed1_adaptiveKL_full`, `20261121-230508-165306_Hopper-v4_seed2_adaptiveKL_full`, `20261121-215526-144460_Walker2d-v4_seed0_adaptiveKL_full`, `20261121-223035-470443_Walker2d-v4_seed1_adaptiveKL_full`, `20261121-230546-934875_Walker2d-v4_seed2_adaptiveKL_full` |
| C1 `noclip` | `20261121-215526-138870_HalfCheetah-v4_seed0_noclip_full`, `20261121-222808-938252_HalfCheetah-v4_seed1_noclip_full`, `20261121-230044-488791_HalfCheetah-v4_seed2_noclip_full`, `20261121-215526-146475_Hopper-v4_seed0_noclip_full`, `20261121-223020-582010_Hopper-v4_seed1_noclip_full`, `20261121-230508-218045_Hopper-v4_seed2_noclip_full`, `20261121-215526-134976_Walker2d-v4_seed0_noclip_full`, `20261121-223056-852731_Walker2d-v4_seed1_noclip_full`, `20261121-230547-167876_Walker2d-v4_seed2_noclip_full` |
| C2 `K=1` | `20261127-155118-732543_HalfCheetah-v4_seed0_clip0.2_k1_full`, `20261127-155118-814204_HalfCheetah-v4_seed1_clip0.2_k1_full`, `20261127-155118-882871_HalfCheetah-v4_seed2_clip0.2_k1_full`, `20261127-155118-984159_Hopper-v4_seed0_clip0.2_k1_full`, `20261127-155119-010752_Hopper-v4_seed1_clip0.2_k1_full`, `20261127-161123-653723_Hopper-v4_seed2_clip0.2_k1_full`, `20261127-161304-334454_Walker2d-v4_seed0_clip0.2_k1_full`, `20261127-162048-095134_Walker2d-v4_seed1_clip0.2_k1_full`, `20261127-162049-256889_Walker2d-v4_seed2_clip0.2_k1_full` |
| C2 `K=10` | `20261127-155118-775521_HalfCheetah-v4_seed0_clip0.2_k10_full`, `20261127-155118-850393_HalfCheetah-v4_seed1_clip0.2_k10_full`, `20261127-155118-917468_HalfCheetah-v4_seed2_clip0.2_k10_full`, `20261127-155118-949549_Hopper-v4_seed0_clip0.2_k10_full`, `20261127-161118-456422_Hopper-v4_seed1_clip0.2_k10_full`, `20261127-161124-809787_Hopper-v4_seed2_clip0.2_k10_full`, `20261127-161313-694009_Walker2d-v4_seed0_clip0.2_k10_full`, `20261127-162048-128655_Walker2d-v4_seed1_clip0.2_k10_full`, `20261127-162224-063132_Walker2d-v4_seed2_clip0.2_k10_full` |

## Regenerate every derived artifact

```powershell
python scripts/make_all.py     # summaries, figures and this table from runs/
```

Machine-readable form: [`logs/paper_vs_ours.json`](../logs/paper_vs_ours.json). Run index: [`logs/runs.csv`](../logs/runs.csv).

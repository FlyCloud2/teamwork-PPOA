# Deviations and measurement definitions

These choices describe the W2 pipeline. This project does not claim bitwise or
absolute-score equivalence with the 2017 PPO paper.

| Item | W2 implementation / implication |
|---|---|
| Environment scale | HalfCheetah-v4, Hopper-v4, Walker2d-v4 only; one seed (0) for W2. This validates the pipeline, not C1 or statistical significance. |
| Simulator | Gymnasium 0.29.1 / MuJoCo 3.1.6, modern v4 environments; historical paper scores are not directly comparable. |
| Wrappers | CleanRL chain: flatten observation, raw episode statistics, clip actions, normalize/clip observations and rewards. Observation/reward clipping stays at [-10, 10]. |
| Value loss | CleanRL clipped value loss retained (`--clip-vloss true`), with advantage normalization per minibatch and global gradient clipping. |
| Time limits | Existing CleanRL behavior retained: both termination and truncation stop GAE bootstrapping. Time-limit bootstrap handling has not been changed in W2. |
| Exact step budget | Old code used floor(1,000,000 / 2,048) = 488 updates and 999,424 steps. W2 runs 488 full batches plus one 576-step batch, exactly 1,000,000 steps. Last batch has 32 minibatches of 18 samples and 10 epochs. Learning-rate annealing spans all 489 updates. This is an explicit departure from the vendored floor-budget loop. |
| Smoke budget | Exactly 20,000 steps: 9 full batches plus 1,568 steps, 10 updates. Smoke data are excluded from W2 acceptance. |
| CPU and threads | CPU, one OMP/MKL thread per W2 process, rather than the old reference setting of four. Fixed across all three runs and recorded. No CUDA benchmark was performed. |
| Episode return column | Most recently completed **raw training episode** at each update boundary, carried forward if no new episode ended. It is not an evaluation average. A trailing-update mean is descriptive, not a mean over seeds or episodes. |
| Diagnostics | Policy/value loss and approx_kl are the final minibatch values; clipfrac is averaged over optimization minibatches. These are not independent evaluation metrics. |
| Variant naming | Legacy `adaptiveKL` was clipped PPO plus target-KL early stopping, not an adaptive KL penalty. That label was rejected during W2; the approximation is available as `kl-early-stop`. In W3 the label `adaptiveKL` was re-used for a real KL penalty, and the legacy behaviour kept the `kl-early-stop` name. |
| noclip limitation | Legacy `noclip` uses epsilon=1e9 and also affects value clipping, since the vendored loss shares epsilon. It is not part of the W2 matrix; isolate the policy objective before a W3 scientific comparison. |
| Provenance | Runs record the existing Git commit, dirty flag and SHA-256 of actual training sources/requirements. W2 was launched from a modified worktree before the implementation commit; source hashes identify the exact executed files. |

The W2 report links raw logs and actual counts. No result is inferred from a
dry-run, a zero process exit code alone, or the requested step count.

## W3 additions (C1, 2026-11-22)

| Item | Original setting | Setting used | Reason / consequence |
|---|---|---|---|
| Adaptive KL implementation | Paper Algorithm 2; the official baselines implementation was not used | Re-implemented inside the vendored trainer: `loss = pg_loss_unclipped + beta * approx_kl`, beta adapted per update, printed to `stdout.log` | The paper's Table 1 middle entry could not be reproduced without an implementation; this one is ours, not the authors'. |
| Beta upper bound | Paper states no upper bound | Default none (`1e300`); two controls at `1/d_targ = 100` and `1` | Added because the literal rule drove beta to 1e150. |
| Beta / gradient clipping interaction | Not specified by the paper | Unchanged CleanRL global clipping at 0.5 | Measured consequence: all three beta settings produce **bit-identical trajectories** (identical `progress.csv` except `wall_time_s`), so beta's magnitude has no effect once the penalty dominates. The `kl-penalty` variant therefore degenerates to "minimise KL" and is not a valid test of the paper's middle position. |
| Seeds per condition | 3 (course requirement) | 3 (0/1/2) | Only directional claims are supported; single-seed variance is large (e.g. clip0.2 on HalfCheetah spans 1247–3098). |
| Python patch version | Teammate machine: 3.10.22 | W3 machine: 3.10.21 | Same ABI and same lock file; recorded here because `env.txt` is not regenerated per machine. |
| W3 hardware | W2: 20-core Intel, 15.8 GiB | W3: Ryzen 7 5800H, 16 logical cores, 15.9 GiB, CPU only | Nine runs ran in parallel, one thread each; wall time per run 32–34 min. |
| W3 package source | PyPI | Tsinghua mirror for the 42 small packages, PyTorch CPU index for torch | The default index was unusable on this machine (~56 KB/s). Versions still come from the pinned lock file, so the resolved set is unchanged. |

The W3 numbers are read from `runs/<id>/progress.csv` by `scripts/plot_c1.py`
and the summary script; see `docs/w3_c1.md`. Re-running the same variants with
the same seeds reproduces the same trajectories.


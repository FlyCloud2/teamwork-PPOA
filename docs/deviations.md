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

## W4 additions (C2 and the Paper-vs-Ours table, 2026-11-27)

| Item | Original setting | Setting used | Reason / consequence |
|---|---|---|---|
| C2 K=10 arm | Paper Table 3 default K = 10 | Produced again on the W4 host by `scripts/run_c2.py`, not reused from W3 | W3 ran on another machine; a host change is a confound the single-factor comparison cannot afford. The W3 `clip0.2` runs remain as an independent cross-host replication. |
| W4 machine | W2: i7-12700H (20 logical, 15.8 GiB). W3: Ryzen 7 5800H (16 logical). | W4: i7-13650HX (14 cores / 20 logical, 31.6 GiB), Windows 11 26200 | Recorded in every `config.json` host block. Wall-clock times are not compared across weeks. |
| PyTorch build | `requirements-lock.txt` pins `torch==2.4.1` (CPU wheel in W2/W3) | `2.4.1+cu121` CUDA build, used on CPU | The version matches the lock; the BLAS/backend differs and is recorded per run. Trajectories are therefore not bit-identical to W3's, which is why C2 is not stitched from W3. |
| Python patch version | W2: 3.10.22, W3: 3.10.21 | W4: 3.10.22 (conda-forge) | Same ABI, same lock file; `env.txt` is not regenerated per machine. |
| Meaning of K = 1 | The paper's K is the number of epochs over the same rollout | K = 1 still performs 32 minibatch gradient steps (batch 2,048, 32 minibatches of 64), not one step | The contrast isolates *repetition of already-seen data*, not "one gradient step per rollout". A one-minibatch ablation is not part of C2. |
| Advantage normalization in the two arms | Per-minibatch normalization, as in W2/W3 | Unchanged in both arms | With K > 1 the same samples are re-normalized every epoch, so the arms differ in repeated gradient steps **and** repeated normalization. This is inseparable from K in this implementation and is stated with the C2 result. |
| Exact step budget | 1,000,000 steps = 488 full batches + one 576-step batch (489 updates) | Same in both arms; the final batch still does K epochs (32 minibatches of 18 at K = 10) | Keeps C2 comparable to C1 and to the frozen W2 budget. |
| Metric | Raw training return of the most recent completed episode at each update; tail-10 mean | Same as C1 | Training returns sampled at update boundaries are not evaluation scores, and 3 seeds support directional claims only. |
| Cost of K | Not reported by the paper | Wall-clock per run is recorded; K = 10 does ten times the minibatch work per update | The W4 matrix ran 9 processes concurrently on one host, so wall time measures relative cost inside this matrix, not a benchmark. |
| Paper-vs-ours normalisation | Paper Table 1: per-task shift-and-scale, random policy = 0, best result = 1, no variance reported | Ours: raw returns, mean ± std over 3 seeds; the "ratio" column divides by the best tier in the same task | We never evaluated an untrained policy, so there is no random-policy zero point. Absolute values are not comparable; the comparison is the ordering, checked per task in `docs/paper_vs_ours.md` Table 3. |
| Task subset in the aggregate | Paper averages 7 MuJoCo tasks | 3 tasks (HalfCheetah, Hopper, Walker2d) | Declared reduced scale; the "ours" aggregate is a 3-task subset, never presented as the paper's 7-task average. |
| Seeds | 3 seeds per setting | 3 (0/1/2) | The paper reports no std; our mean ± std is an addition, not a reproduction of a published interval. |
| W2 baseline source hashes | W2 recorded the SHA-256 of the sources it executed, and `verify_runs.py --baseline` compared them with the current files | W3 added the KL-penalty flags to `src/ppo_args.py` / `src/ppo_continuous_action.py`, so from W3 onward that comparison failed | Fixed in W4: `--baseline` now requires the three W2 runs to agree with each other and **reports** drift against the current tree instead of failing. The executed revision stays identified by the recorded hashes and commit. Until W4 this also made `weekly_check.ps1` step 2 fail. |

The W4 numbers are read from `runs/<id>/progress.csv` by `scripts/summarize_c2.py`,
`scripts/plot_c2.py` and `scripts/make_table.py`; see `docs/w4_c2.md` and the
generated `docs/paper_vs_ours.md`.


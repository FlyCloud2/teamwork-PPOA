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
| Variant naming | Legacy `adaptiveKL` was clipped PPO plus target-KL early stopping, not an adaptive KL penalty. It now fails explicitly. The approximation is available as `kl-early-stop`; actual adaptive-KL implementation remains W3 work. |
| noclip limitation | Legacy `noclip` uses epsilon=1e9 and also affects value clipping, since the vendored loss shares epsilon. It is not part of the W2 matrix; isolate the policy objective before a W3 scientific comparison. |
| Provenance | Runs record the existing Git commit, dirty flag and SHA-256 of actual training sources/requirements. W2 was launched from a modified worktree before the implementation commit; source hashes identify the exact executed files. |

The W2 report links raw logs and actual counts. No result is inferred from a
dry-run, a zero process exit code alone, or the requested step count.

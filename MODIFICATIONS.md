# MODIFICATIONS -- changes to the vendored CleanRL trainer

## Pinned upstream

- Repository: https://github.com/vwxyzjn/cleanrl
- Commit: `cf20043b852deedf52dc9f90a325c6866d599005` (2023-10-15).
- Actual commit subject: `Reducing Code Redundancy and Minimize Line Diffs (#422)`.
  The previous document incorrectly associated this SHA with a Hugging Face change.
- Upstream file: `cleanrl/ppo_continuous_action.py`.
- Local file: `src/ppo_continuous_action.py`.
- License: MIT, retained in `LICENSES/cleanrl-LICENSE.txt`.

The user identified this upstream repository on 2026-11-15. The commit and source
were re-fetched and checked against it; the current upstream default branch was
not substituted for the pinned implementation.

## W1 modifications retained

The local copy adds a Chinese module description and translated comments,
`--csv-out` and `--tb-out`, a per-update CSV writer, explicit TensorBoard output
paths, and the most recent completed episode's raw return/length carried into
the next CSV row. CSV output is flushed after each update and closed at exit.
The core loss formulas were retained. Historical `<name>` / `<date>` comments
were unfilled templates, not verified human sign-offs; see `AI_USAGE.md`.

The old document claimed whole-file AST identity despite these logging additions.
That claim and its non-working PowerShell/Bash mixed command have been removed.
The scoped audit below states exactly which parts were compared.

## W2 changes (2026-11-15)

1. Extract argparse into `src/ppo_args.py`, shared by runner and trainer. Remove
   `distutils`, validate batch arithmetic and numerical inputs, and explicitly
   forward exposed training parameters. CPU is the default; unavailable requested
   CUDA raises an error. Device and thread count are printed into the run log.
2. Expose hidden layer widths, activation and Adam epsilon, retaining the default
   two 64-unit tanh layers and their initialization order.
3. Use a shorter final rollout to consume the exact requested budget. GAE and
   flattened training tensors use only that rollout's valid prefix; the last
   minibatch size derives from the actual batch. A million steps now requires
   489 updates, with a final 576-sample batch (32 minibatches of 18). Learning-rate
   annealing spans 489 updates. This intentionally differs from floor-budget
   CleanRL; see `docs/deviations.md`.
4. Add runner source hashes, dependency versions, running/completed/failed status,
   step validation and a locked, atomically replaced CSV index. Use unique run IDs
   and UTF-8 child output; preserve failures without counting them as baselines.
5. Reject the misleading legacy `adaptiveKL` label: its old implementation was
   clipped PPO with KL early stopping, not the paper's adaptive penalty. The old
   behavior is named `kl-early-stop`; the real adaptive penalty remains W3 work.

The trainer's opening W1 description of unchanged update code is historical;
the explicit final-rollout changes above are part of the executed W2 source.
Source hashes are kept unchanged after the full runs to retain their provenance.

## W3 changes (2026-11-22)

1. Add a policy-surrogate switch to the shared schema: `--surrogate clip|kl-penalty`
   (default `clip`), plus `--kl-coef` (initial beta, default 1.0), `--kl-target`
   (d_targ, default 0.01) and `--kl-coef-max` (upper bound, default `1e300`,
   i.e. effectively none). Validation rejects a non-positive beta or target, and
   rejects `kl-penalty` combined with `--target-kl`, because that pair would mix
   the penalty with the early-stopping rule.
2. Implement the KL-penalty surrogate of the paper's Section 4. The policy term
   is the unclipped objective and the total loss adds `beta * approx_kl`, where
   `approx_kl` is the same k3 estimator the CSV already records
   (`((ratio - 1) - logratio).mean()`).
3. Implement the beta adaptation of Algorithm 2: after each policy update, take
   the mean KL over that update's minibatches; if it is below `d_targ / 1.5`
   halve beta, if above `d_targ * 1.5` double it; then clamp to `--kl-coef-max`.
   Every change is printed to the run's `stdout.log`, so the beta trajectory is
   traceable without a new CSV column. The CSV schema is unchanged.
4. Only the policy surrogate changes. `--clip-coef`, value clipping, GAE,
   normalization, seeds and the step budget stay identical to `clip0.2`, so the
   C1 comparison differs in one factor. `clipfrac` keeps meaning "fraction of
   ratios outside the 0.2 band" and is therefore a diagnostic, not a loss term,
   under `kl-penalty`.
5. The regression test that asserted `adaptiveKL` must be rejected is replaced:
   an unknown variant name now fills that role, and three tests pin the new
   mapping (`adaptiveKL`, `adaptiveKL-capped`, `adaptiveKL-tight`) including the
   requirement that they differ only in `kl_coef_max`.

## W4 changes (2026-11-27)

C2 changes **no training code**. `update_epochs` has been a first-class CLI
parameter since W2, so both arms run the same vendored trainer and the same
loss; the only runner-side change is a label suffix so that two profiles of the
same variant cannot be confused in `runs/`:

1. `scripts/run.py --label`: validated to `[A-Za-z0-9][A-Za-z0-9.-]{0,23}`,
   written into the run directory name (`..._clip0.2_k1_full`) and into
   `config.json` as `label`. It is **not** forwarded to the trainer, so it cannot
   change a trajectory.
2. `scripts/verify_runs.py --c2`: matrix verification that the two arms differ
   in `update_epochs` and in nothing else (environments, seeds, threads, package
   set and platform are checked separately), plus the requirement that the run id
   encodes K.
3. New evidence scripts, none of which is imported by the trainer:
   `evidence.py` (run selection and aggregation), `run_c2.py` (18-run matrix),
   `summarize_c2.py`, `plot_c2.py`, `make_table.py` (Paper-vs-Ours tables) and
   `make_all.py` (regenerate every artifact).

`src/ppo_args.py` and `src/ppo_continuous_action.py` are unchanged since W3.
Every W4 run records their SHA-256 in `config.json`; the values match the
delivered files, so the executed trainer is the reviewed one.

## Reproduce the upstream audit

Run from the repository root with the tested Python environment:

```powershell
$UpstreamPpoUrl = 'https://raw.githubusercontent.com/vwxyzjn/cleanrl/cf20043b852deedf52dc9f90a325c6866d599005/cleanrl/ppo_continuous_action.py'
Invoke-WebRequest $UpstreamPpoUrl -OutFile .git/cleanrl_upstream.py
python scripts/audit_upstream.py --upstream .git/cleanrl_upstream.py
```

The generated `logs/upstream_audit.json` records source hashes and seven checks:
`make_env`, `layer_init`, `get_value`, `get_action_and_value`, the minibatch loss
and optimizer body, the GAE recurrence, and bitwise equality of the default
network's initialized parameters at seed 0. All seven passed. This is a scoped
AST/numerical audit, not a claim of whole-file identity or bitwise equality of
training trajectories. Run-local configuration/source hashes and the deviations
table define the W2 execution more precisely.

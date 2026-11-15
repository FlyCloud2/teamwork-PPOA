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

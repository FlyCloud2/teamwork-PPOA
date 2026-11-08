# MODIFICATIONS -- what we changed in the vendored CleanRL file

## Starting point

| | |
|---|---|
| Upstream repository | <https://github.com/vwxyzjn/cleanrl> |
| Commit | `cf20043b852deedf52dc9f90a325c6866d599005` (2023-10-15, "ppo_continuous_action huggingface integration (#423)") |
| File | `cleanrl/ppo_continuous_action.py` |
| Local copy | `src/ppo_continuous_action.py` |
| Licence | MIT -- `LICENSES/cleanrl-LICENSE.txt` |
| Retrieved | 2026-10-30 |

The commit is pinned, not floating: `pip install cleanrl` or a submodule pointing at
`master` would let the algorithm change under our results, which breaks the
reproduction claim. A vendored single file is frozen the moment we commit it.

## The 8 changes

Every edit below carries a `[REPRO]` comment in the source. Nothing inside the
policy/value computation was touched -- the loss, the ratio, the clipping and the GAE
recursion are byte-identical to upstream.

Audit note for change 8: the whole file's comments and docstrings were translated to
Chinese. To prove that no executable line moved, the AST of this file with docstrings
stripped was compared against the pristine upstream copy -- the two are identical
(`ast.dump` equality, see "How to re-derive this diff"). Comments do not appear in an
AST at all, so that check isolates exactly what change 8 touched.

### 1. Module docstring (lines 2-37)

Upstream has no module docstring. Ours states what the file is, the exact upstream
commit, how to invoke it by hand, and what was changed -- so a reader who opens
`src/` first is not left guessing whether this is "CleanRL" or "our PPO".

### 2. Two extra imports

```diff
+# [REPRO] +csv/pathlib: needed by the CSV evidence trail below (otherwise unused upstream).
+import csv
+import pathlib
 import random
```

### 3. Two output arguments

```diff
+    # [REPRO] Output destinations. Both default to None so the upstream behaviour
+    # (tensorboard into ./runs/<run_name>) is unchanged when they are not passed.
+    parser.add_argument("--csv-out", type=str, default=None,
+        help="path of the per-update progress CSV; the only durable evidence trail")
+    parser.add_argument("--tb-out", type=str, default=None,
+        help="directory for tensorboard event files (default: runs/<run_name>)")
```

### 4. Tensorboard events go into the run directory

```diff
-    writer = SummaryWriter(f"runs/{run_name}")
+    writer = SummaryWriter(args.tb_out if args.tb_out else f"runs/{run_name}")
```

Why: event files are binary, large and unreadable in a diff, so they are gitignored.
Upstream writes them to `./runs/` relative to the working directory; with a runner
that means a stray `runs/` folder in the repository root. The runner now passes
`runs/<id>/tensorboard`.

### 5. CSV writer initialised before the training loop

```diff
+    csv_file, csv_writer = None, None
+    if args.csv_out is not None:
+        pathlib.Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
+        csv_file = open(args.csv_out, "w", newline="")
+        csv_writer = csv.writer(csv_file)
+        csv_writer.writerow([... 11 column names ...])
+        csv_file.flush()
+    last_ep_ret, last_ep_len = float("nan"), float("nan")
```

Why: the report may not quote a number that cannot be traced to a committed log.
Upstream only writes tensorboard events, which are unusable for a table or a diff.
The row is flushed on every update so a crashed run still leaves its partial evidence.

### 6. Latest finished-episode statistics are carried forward

```diff
             for info in infos["final_info"]:
                 if info is None:
                     continue
                 print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                 writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                 writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
+                last_ep_ret = float(info["episode"]["r"])
+                last_ep_len = float(info["episode"]["l"])
```

Why: an episode ends roughly every 1000 environment steps while an update covers
2048, so an episode almost never ends exactly on an update boundary. Upstream only
prints at the moment of finishing; the CSV needs the most recent values stamped onto
the next row.

### 7. One CSV row per update, plus a final close

```diff
+        if csv_writer is not None:
+            csv_writer.writerow([global_step, last_ep_ret, last_ep_len,
+                                 pg_loss.item(), v_loss.item(), entropy_loss.item(),
+                                 approx_kl.item(), float(np.mean(clipfracs)),
+                                 explained_var, optimizer.param_groups[0]["lr"],
+                                 round(time.time() - start_time, 3)])
+            csv_file.flush()
...
+    if csv_file is not None:
+        csv_file.close()
```

The column set is fixed for the whole project (`configs/base.yaml: logging.columns`).
`approx_kl`, `clipfrac` and `explained_variance` look like diagnostics, but they are
the only direct evidence for the W6 critique "is the clipping mechanism actually
doing anything" -- adding them later would mean re-running every experiment.

### 8. Comments and docstrings translated to Chinese

The team works in Chinese, so every comment in this file -- upstream prose included --
was translated. No code was touched: identifiers, string literals, CLI flags and the
`# fmt: off/on` pragmas are unchanged, and the AST (docstrings stripped) is identical
to upstream.

Two exceptions are deliberate, because they are not prose:

* `# fmt: off` / `# fmt: on` -- directives for black; translating them would disable the
  formatter.
* `# docs and experiment results can be found at <url>` -- a pointer; the URL is kept
  byte-exact and only the leading words are translated.

All 8 changes, in the order a reader meets them: docstring, imports, output arguments,
tensorboard target, CSV init, episode-stats carry-forward, per-update row + close, and
this translation pass.

## Also added (not edits to upstream code)

* Three explanatory comments inside the training loop, marking eq.(7), eq.(10) of
  arXiv:1707.06347v2 next to the lines that implement them, and one comment on the
  2-argument `TransformObservation` call explaining the `gymnasium==0.29.1` pin.
  These are comments only -- no executable statement changed.

## How to re-derive this diff

```powershell
# 1) fetch the exact upstream file
$url = "https://raw.githubusercontent.com/vwxyzjn/cleanrl/cf20043b852deedf52dc9f90a325c6866d599005/cleanrl/ppo_continuous_action.py"
Invoke-WebRequest $url -OutFile upstream_ppo.py

# 2) diff it against our copy (git is required; it ships a diff tool)
git diff --no-index -- upstream_ppo.py src/ppo_continuous_action.py

# 3) prove that only comments and docstrings differ, i.e. no executable line moved
python - upstream_ppo.py src/ppo_continuous_action.py <<'PY'
import ast, sys

def strip(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                 and isinstance(b[0].value.value, str):
                n.body = b[1:]
    return tree

a = strip(ast.parse(open(sys.argv[1], encoding="utf-8").read()))
b = strip(ast.parse(open(sys.argv[2], encoding="utf-8").read()))
print("AST identical:", ast.dump(a) == ast.dump(b))
PY
```

Without step 3, change 8 turns the diff into hundreds of noise lines. With it, the
verification is mechanical: the AST equal, and every hunk being comment-only, means the
algorithm is upstream's.

Everything the diff shows must be one of the 8 items above. If it is not, either the
change is undocumented or the file drifted -- both are bugs.

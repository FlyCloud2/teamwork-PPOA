#!/usr/bin/env python
"""Run PPO with explicit parameters, exact step budgets and auditable artifacts."""
import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ppo_args import build_parser, validate_args

PPO = REPO / "src" / "ppo_continuous_action.py"
RUNS = REPO / "runs"
LOGS = REPO / "logs"
VARIANTS = ("clip0.2", "noclip", "kl-early-stop", "custom", "adaptiveKL",
            "adaptiveKL-capped", "adaptiveKL-tight")
INDEX_FIELDS = ["run_id", "started", "env", "seed", "variant", "steps", "actual_steps",
                "updates", "status", "wall_time_s", "return_code", "git_commit_short"]


def git(*args, default="unknown"):
    try:
        return subprocess.check_output([shutil.which("git") or "git", *args], cwd=REPO,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def atomic_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


@contextlib.contextmanager
def index_lock(path):
    """OS-owned lock: concurrent runners cannot overwrite each other's index rows."""
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def update_index(config, root=REPO):
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "runs.csv"
    with index_lock(logs / ".runs.lock"):
        rows = []
        if path.exists():
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        rows = [row for row in rows if row["run_id"] != config["run_id"]]
        rows.append({key: config.get(key, "") for key in INDEX_FIELDS})
        rows.sort(key=lambda row: row["run_id"])
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def inspect_progress(path, hp):
    """Require every expected update, exact steps and finite optimization metrics."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = list(range(hp["batch_size"], hp["total_timesteps"], hp["batch_size"]))
    expected.append(hp["total_timesteps"])
    actual = [int(row["global_step"]) for row in rows]
    if actual != expected:
        raise ValueError(f"incomplete step sequence: {len(actual)} updates, expected {len(expected)}")
    metrics = ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "learning_rate", "wall_time_s")
    for row in rows:
        for key in metrics:
            if not math.isfinite(float(row[key])):
                raise ValueError(f"non-finite {key} at step {row['global_step']}")
    return {"actual_steps": actual[-1], "updates": len(rows)}


def parse_runner_args(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--env", dest="env_id", default=argparse.SUPPRESS, help="alias for --env-id")
    parser.add_argument("--variant", choices=VARIANTS, default="clip0.2")
    # [REPRO] W4：同一 variant 下还要做单因子对照（例如 update_epochs=1 vs 10）。
    # run_id 只由「时间 / 环境 / 种子 / variant / 类型」拼成，两个 profile 会撞名，
    # 于是用 --label 显式写进 run_id：..._clip0.2_k1_full。它不传给训练器，
    # 只是运行档案的标签，同时写进 config.json 以便溯源。
    parser.add_argument("--label", default="",
                        help="short suffix that distinguishes otherwise identical runs (e.g. k1); written into run_id and config.json")
    parser.add_argument("--smoke", action="store_true", help="20,000 steps; never baseline evidence")
    parser.add_argument("--threads", type=int, default=4, help="fixed OMP/MKL thread count")
    parser.add_argument("--dry-run", action="store_true", help="print resolved configuration without writing files")
    parser.add_argument("--output-root", type=Path, default=REPO, help="root containing runs/ and logs/")
    parser.set_defaults(seed=0)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    # A label becomes part of a directory name: allow only characters that cannot
    # introduce a path separator, a glob wildcard or an ambiguous run_id.
    if args.label and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,23}", args.label):
        parser.error("--label must be 1-24 characters of letters, digits, '.' or '-', starting alphanumeric")
    if args.variant in ("adaptiveKL", "adaptiveKL-capped", "adaptiveKL-tight"):
        # [REPRO] W3：论文 Table 1 的第二种代理目标 —— 自适应 KL 罚项。
        # 只切换策略代理目标，其余（种子、超参、步数预算、价值裁剪）与 clip0.2 完全一致，
        # 这样 C1 里两者的差异只能归因到代理目标本身。
        # 目标 KL 取论文里表现居中的 d_targ = 0.01。
        args.surrogate = "kl-penalty"
        args.kl_coef = 1.0
        args.kl_target = 0.01
        args.target_kl = None
        if args.variant == "adaptiveKL-capped":
            # 受控对照一：上界 1 / d_targ。实测与字面版逐行相同 —— 梯度裁剪把 beta 的大小抹平了。
            args.kl_coef_max = 1.0 / args.kl_target
        elif args.variant == "adaptiveKL-tight":
            # 受控对照二：上界压到 1，即 beta 永远不高于初始值。
            # 用来区分「KL 罚项本身不行」与「beta 的调整规则太激进」。
            args.kl_coef_max = 1.0
        else:
            # 字面版：不设上界，展示 Algorithm 2 的原始行为。
            args.kl_coef_max = 1e300
    if args.variant == "clip0.2" and (args.clip_coef != 0.2 or args.target_kl is not None):
        parser.error("clip0.2 requires --clip-coef 0.2 and no --target-kl; use --variant custom")
    if args.variant == "noclip":
        args.clip_coef = 1e9
    if args.variant == "kl-early-stop":
        args.target_kl = 0.01 if args.target_kl is None else args.target_kl
    if args.smoke:
        args.total_timesteps = 20_000
    # Output paths belong to the runner, not independent caller-supplied destinations.
    if args.csv_out or args.tb_out:
        parser.error("use --output-root; the runner owns --csv-out and --tb-out")
    return validate_args(args, parser)


def child_command(args, out):
    trainer = build_parser()
    values = vars(args).copy()
    values.update(exp_name=args.variant, csv_out=str(out / "progress.csv"), tb_out=str(out / "tensorboard"))
    cmd = [sys.executable, "-u", str(PPO)]
    hp = {}
    for action in trainer._actions:
        if action.dest == "help":
            continue
        value = values[action.dest]
        hp[action.dest] = value
        if value is None:
            continue
        items = value if isinstance(value, list) else [value]
        cmd.extend([action.option_strings[0], *[str(item).lower() if isinstance(item, bool) else str(item) for item in items]])
    hp.update(batch_size=args.batch_size, minibatch_size=args.minibatch_size, num_updates=args.num_updates)
    return cmd, hp


def run_id(env_id, seed, variant, label="", smoke=False, now=None):
    """Unique, self-describing run directory name."""
    tag = "smoke" if smoke else "full"
    rid = f"{now or dt.datetime.now():%Y%m%d-%H%M%S-%f}_{env_id}_seed{seed}_{variant}"
    if label:
        rid += f"_{label}"
    return f"{rid}_{tag}"


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_runner_args(argv)
    rid = run_id(args.env_id, args.seed, args.variant, args.label, args.smoke)
    # Environment IDs can include namespaces; never allow them to create nested output paths.
    rid = rid.replace("/", "_").replace("\\", "_")
    root = args.output_root.resolve()
    out = root / "runs" / rid
    cmd, hp = child_command(args, out)
    if args.dry_run:
        print(json.dumps({"argv": cmd, "hyperparameters": hp, "threads": args.threads}, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    sources = ["src/ppo_args.py", "src/ppo_continuous_action.py", "scripts/run.py", "requirements.txt"]
    packages = {}
    for name in ("torch", "gymnasium", "mujoco", "numpy", "tensorboard"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = "NOT INSTALLED"
    config = {
        "schema_version": 2, "run_id": rid, "argv": cmd,
        "git_commit": git("rev-parse", "HEAD"), "git_commit_short": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", default="")),
        "source_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in sources},
        "variant": args.variant, "seed": args.seed, "steps": args.total_timesteps,
        "label": args.label,
        "env": args.env_id, "threads": args.threads, "smoke": args.smoke,
        "hyperparameters": hp, "packages": packages,
        "host": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()},
        "started": dt.datetime.now().isoformat(timespec="seconds"), "status": "running",
    }
    atomic_json(out / "config.json", config)
    update_index(config, root)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    t0 = time.monotonic()
    rc = 1
    print(f"[run] started {rid}", flush=True)
    try:
        with (out / "stdout.log").open("wb") as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=env)
            try:
                rc = child.wait()
            except KeyboardInterrupt:
                child.terminate()
                child.wait()
                rc = 130
        if rc == 0:
            config.update(inspect_progress(out / "progress.csv", hp))
        config["status"] = "completed" if rc == 0 else ("interrupted" if rc == 130 else "failed")
    except (OSError, ValueError, KeyError) as exc:
        rc = 1
        config.update(status="failed", error=str(exc))
    finally:
        config.update(finished=dt.datetime.now().isoformat(timespec="seconds"),
                      wall_time_s=round(time.monotonic() - t0, 3), return_code=rc)
        atomic_json(out / "config.json", config)
        update_index(config, root)
    print(f"[run] {rid} rc={rc} {config['wall_time_s']:.1f}s -> {out}")
    if rc:
        print(config.get("error", ""))
        if (out / "stdout.log").exists():
            print("\n".join((out / "stdout.log").read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
    return rc


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""统一 runner：一条命令进，一个自解释的 run 目录出。

为什么要有它
    报告里的一个数字，除非陌生人只靠仓库就能回答三个问题，否则一文不值：
    它是哪个 commit 跑出来的、用了哪套超参、
    出自哪份日志。本脚本是唯一会启动训练的入口，
    而它把这三个答案写在数字旁边。

它写出什么
    runs/<run_id>/config.json    命令行、解析后的超参、git commit、时间
    runs/<run_id>/progress.csv   每次更新的指标（由 src/ppo_continuous_action.py 产生）
    runs/<run_id>/stdout.log     子进程的原始控制台输出
    logs/runs.csv                每次 run 一行索引（一眼看全实验矩阵）

怎么运行
    python scripts/run.py --env HalfCheetah-v4 --seed 0 --variant clip0.2 --smoke
    python scripts/run.py --env Hopper-v4 --seed 1 --variant noclip           # 完整 100 万步
    python scripts/run.py --env Walker2d-v4 --seed 2 --variant clip0.2 --dry-run

示例（冒烟，约 1 分钟）
    上面第一条命令加上 --smoke，在 env.txt 记录的机器上约 1 分钟跑完，
    留下 runs/<时间戳>_HalfCheetah-v4_seed0_clip0.2_smoke/。
"""
import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
# 来源标注：AI 辅助起草 —— 见 AI_USAGE.md。由 <name> 于 <date> 复核。
from pathlib import Path

# 不硬编码绝对路径：一切都由本文件的位置推出，
# 所以换一台机器 clone 下来照样能用。
REPO = Path(__file__).resolve().parents[1]
PPO = REPO / "src" / "ppo_continuous_action.py"
LOGS = REPO / "logs"
RUNS = REPO / "runs"

# 变体名 -> 传给 vendored 训练器的额外开关。
#   clip0.2    ：论文 Table 1 里的 PPO（epsilon = 0.2）
#   noclip     ：epsilon -> 无穷大，即裁剪目标退化成普通策略梯度
#   adaptiveKL ：论文的另一种代理目标（KL 罚项，d_targ = 0.01），
#                用「approx_kl 超过目标就提前结束 epoch」来近似
VARIANTS = {
    "clip0.2": ["--clip-coef", "0.2"],
    "noclip": ["--clip-coef", "1000000000.0"],
    "adaptiveKL": ["--clip-coef", "0.2", "--target-kl", "0.01"],
}


def git(*args, default="unknown"):
    """Run a read-only git command, tolerating a machine where git is not on PATH."""
    exe = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
    try:
        out = subprocess.check_output([exe, *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="HalfCheetah-v4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="clip0.2", choices=sorted(VARIANTS))
    ap.add_argument("--smoke", action="store_true", help="20k steps, ~30 s: proves the pipeline, not the science")
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=4, help="OMP/MKL threads; pinned for numeric reproducibility, not speed")
    ap.add_argument("--dry-run", action="store_true", help="print the child command and exit")
    a = ap.parse_args()

    steps = 20_000 if a.smoke else a.total_timesteps
    tag = "smoke" if a.smoke else "full"
    rid = f"{dt.datetime.now():%Y%m%d-%H%M}_{a.env}_seed{a.seed}_{a.variant}_{tag}"
    out = RUNS / rid

    cmd = [
        sys.executable, str(PPO),
        "--env-id", a.env,
        "--seed", str(a.seed),
        "--total-timesteps", str(steps),
        "--exp-name", a.variant,
        "--csv-out", str(out / "progress.csv"),
        "--tb-out", str(out / "tensorboard"),
        *VARIANTS[a.variant],
    ]

    # 在 mkdir 之前就判断：试运行不能留下一个空的 run 目录。
    # 一旦 logs/runs.csv 被当成「真实发生过的 run」，空目录就是一句假话。
    if a.dry_run:
        print(" ".join(cmd))
        return 0

    out.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": rid,
        "argv": cmd,
        "git_commit": git("rev-parse", "HEAD"),
        "git_commit_short": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", default="")),
        "variant": a.variant,
        "seed": a.seed,
        "steps": steps,
        "env": a.env,
        "threads": a.threads,
        # 重建这个数字所需的一切，不用去读我们的代码。
        "hyperparameters": {
            "num_envs": 1, "num_steps": 2048, "num_minibatches": 32, "update_epochs": 10,
            "gamma": 0.99, "gae_lambda": 0.95, "learning_rate": 3e-4, "anneal_lr": True,
            "clip_coef": 1e9 if a.variant == "noclip" else 0.2,
            "target_kl": 0.01 if a.variant == "adaptiveKL" else None,
            "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
            "clip_vloss": True, "norm_adv": True,
            "hidden_sizes": [64, 64], "activation": "tanh",
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "started": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8", newline="\n")

    # 线程数是钉住的，不是调出来的。在参考机器上 A/B 没有可测的速度差异
    # （167 vs 165 SPS，属噪声），但 OMP 线程数会改变浮点归约顺序，
    # 所以它首先是复现变量：
    # 所有 run 用同一个值，并记进 config.json。
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))

    # stdout 写进文件、绝不走管道：子进程跑在中文代码页的控制台里，
    # 管道解码会抛 UnicodeDecodeError（docs/setup.md 坑 #7）。
    t0 = time.time()
    with open(out / "stdout.log", "wb") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO), env=env)
    elapsed = time.time() - t0

    config["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    config["wall_time_s"] = round(elapsed, 3)
    config["return_code"] = rc
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8", newline="\n")

    LOGS.mkdir(exist_ok=True)
    index = LOGS / "runs.csv"
    new = not index.exists()
    with open(index, "a", newline="\n", encoding="utf-8") as f:
        if new:
            f.write("run_id,started,env,seed,variant,steps,wall_time_s,return_code,git_commit_short\n")
        f.write(f"{rid},{config['started']},{a.env},{a.seed},{a.variant},{steps},"
                f"{config['wall_time_s']},{rc},{config['git_commit_short']}\n")

    print(f"[run] {rid}  rc={rc}  {elapsed:.1f}s  -> {out.relative_to(REPO)}")
    if rc != 0:
        print(f"[run] tail of stdout.log:\n" + "\n".join(
            (out / "stdout.log").read_text(errors="replace").splitlines()[-15:]))
    return rc


if __name__ == "__main__":
    sys.exit(main())

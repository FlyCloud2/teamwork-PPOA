# MIAE01 论文复现：MuJoCo 上的 PPO

课程项目（teamwork-PPOA）。我们从论文本身复现近端策略优化（Proximal Policy Optimization，arXiv:1707.06347v2），而不是调用框架里现成的 `PPO()`；并且检验论文关于"三种代理目标里哪一种真正有效"的核心主张，能否在我们自己的测量中成立。

> **状态：第 3 周（11/16 – 11/22），C1 主对照已完成。** 3 环境 × 3 种子 × 1,000,000 步的
> `clip0.2` / `adaptiveKL` / `noclip` 共 27 次已完成，另有 18 次受控对照
> （`adaptiveKL-capped`、`adaptiveKL-tight`）。裁决是**部分复现**：裁剪优于两端成立，
> 中间那一档在我们的实现下退化、无法判定。结果与原始证据见 [`docs/w3_c1.md`](docs/w3_c1.md)，
> 基线见 [`docs/w2_baseline.md`](docs/w2_baseline.md)。英文版见 [`docs/README.en.md`](docs/README.en.md)。

## 论文主张是什么，我们检验什么

PPO 的贡献是一族代理目标：它不求解带约束的优化问题，也能把每次策略更新限制在信任域附近。论文在 MuJoCo 上比较了其中三种（Table 1）——ε = 0.2 的**裁剪目标**、带目标 KL 的**自适应 KL 罚项**、以及**完全不裁剪**。它主张的是一个**排序**而不是某个具体数字：裁剪 > KL 罚项 > 不裁剪。

我们要尝试推翻的就是这个排序，分三步：

| 编号 | 问题 | 实验配置 |
|---|---|---|
| C1 | 这个排序成立吗？ | `clip0.2` vs `adaptiveKL` vs `noclip`，3 环境 × 3 种子 |
| C2 | 同一个 batch 重复训练多个 epoch，到底买到了什么？ | `update_epochs = 1` vs `10` |
| C3 | 论文从未调过的那个旋钮（K，即 epoch 数）是不是关键？ | `K` 扫描 |

所有比较都以论文自己的机制为基线，所以"复现失败"会表现为排序不同，而不是"我们的数字更低"。

## 范围：我们刻意不做的事

* **只做论文 7 个 MuJoCo 环境中的 3 个**（HalfCheetah、Hopper、Walker2d）。明确声明为 reduced scale，报告里不会把我们的格子与论文的格子逐格对比。
* **绝对分数与论文不可比**：环境 wrapper、MuJoCo 版本、框架都不同，每一处差异都写进 [`docs/deviations.md`](docs/deviations.md)，而且是在做那个实验之前就写。
* **不声称 bit-for-bit 复现**。每个配置 3 个种子，报告 mean ± std，解读的是排序，不是末位数字。

## 七周路线图

一周一个交付物。每个检查点都是硬截止（周日 23:59），且每位成员在截止前推自己的实质 commit。

| 周次 | 日期（检查点） | 目标 | 交付物 |
|---|---|---|---|
| **W1** | 11/02 – 11/08 | 仓库地基、环境钉版本、管线跑通 | `env.txt`、一次冒烟跑的 `progress.csv` + `config.json` + `stdout.log`、第一张曲线、偏差表起头 |
| W2 | 11/09 – 11/15 | baseline 打通到全尺度 | 3 环境 × seed 0 × 100 万步；`logs/runs.csv` 索引；所有参数改成命令行开关；冻结 gymnasium / MuJoCo 版本 |
| W3 | 11/16 – 11/22 | **C1：论文的主 claim** | `clip0.2` / `adaptiveKL` / `noclip` × 3 环境 × 3 种子 = 27 次；每个变体只是一个开关，不改源码 |
| W4 | 11/23 – 11/29 | **C2** + 自动出表 | `update_epochs` 1 与 10 的对照；`scripts/make_table.py` 自动生成 Paper vs Ours 表；报告开始动笔 |
| W5 | 11/30 – 12/06 | Extension（单因子） | `K` 扫描 × 3 环境 × 3 种子，一张图，一句能站住的结论 |
| W6 | 12/07 – 12/13 | 差异归因 + 初稿 | 至少一次受控重跑作为差异证据；偏差表定稿；报告初稿齐全 |
| W7 | 12/14 – 12/20 | 定稿 | 报告 PDF、展示、`v1.0` tag；由非本组同学照 README 从零走通一次 |

> 更细的按周执行清单（本周待办、每周例行、里程碑）见 [`ROADMAP.md`](ROADMAP.md)。

> 日期按"整体顺延一周"重排，第 1 周从 11/02 起算；每周窗口仍是周一 00:00 – 周日 23:59。若课程的展示日仍固定在 12/12，则最后两周（W6 / W7）需在 12/07 – 12/12 内合并完成，以课程通知为准。

## 快速开始

需要 Python 3.10。直接依赖钉在 [`requirements.txt`](requirements.txt)，本次实测的完整依赖冻结在 [`requirements-lock.txt`](requirements-lock.txt)，机器信息见 [`env.txt`](env.txt)。Windows 上 MuJoCo 的虚拟环境必须放在纯英文路径，真实安装过程见 [`docs/setup.md`](docs/setup.md)。

```powershell
python -m pip install -r requirements-lock.txt

python scripts/run.py --env HalfCheetah-v4 --seed 0 --variant clip0.2 --smoke --threads 1
python scripts/plot.py --run runs/<run_id> --out figures/smoke_HalfCheetah.png

# W2：三个独立进程并行；资源有限时用 --jobs 1
python scripts/run_baseline.py --jobs 3 --threads 1
python scripts/verify_runs.py --baseline
python scripts/plot.py --baseline --out figures/w2_baseline.png
python scripts/summarize_baseline.py

# 回归检查（含真实短训练、失败处理和并发索引）
python -m unittest discover -s tests -v
```

**预期输出与运行时长**

| 命令 | 步数 | 预期耗时 | 产出 |
|---|---|---|---|
| `run.py --smoke` | 20,000（10 次更新） | 本机约 91 秒（含启动） | `runs/<id>/progress.csv`（10 行）、`config.json`、`stdout.log` |
| `run.py`（完整） | 精确 1,000,000 | 以每个 run 的 `wall_time_s` 为准 | 489 次更新，末批 576 步；每次一行 |

`scripts/run.py --help` 展示全部训练参数，例如 `--learning-rate`、`--num-envs`、`--num-steps`、`--num-minibatches`、`--update-epochs`、`--gamma`、`--gae-lambda`、`--hidden-sizes`、`--activation`、`--adam-eps`，以及 `--anneal-lr false` 等布尔开关。参数由 runner 和训练器共享同一解析定义；`config.json` 保存实际传入值、实际步数、源码 SHA-256、依赖版本与状态。`--dry-run` 只打印配置，不创建运行证据。

默认使用 CPU；GPU 需显式 `--cuda true`。单次 runner 的线程默认值保留 4，W2 矩阵入口固定默认为 1。`--output-root` 可隔离试验目录。非法批大小、非有限超参数与不足以做优势归一化的末批会在训练前报错。

**变体边界：** `clip0.2`、`noclip`、`kl-early-stop`、`custom` 之外，W3 新增了三个 KL 罚项变体：
`adaptiveKL`（论文 Algorithm 2 的字面实现，β 无上界）、`adaptiveKL-capped`（β ≤ 1/d_targ = 100）、
`adaptiveKL-tight`（β ≤ 1）。实测三者训练轨迹逐行相同——β 主导后梯度裁剪把步长归一化，
β 的具体数值不再影响更新；因此在我们的设置里这一档**无法作为论文中间位置的检验**，
详见 [`docs/w3_c1.md`](docs/w3_c1.md) §3。非默认裁剪系数使用 `--variant custom --clip-coef ...`。

验收以 `logs/w2_baseline.json` 指定的三个完整 run 为准，不能仅看返回码或请求步数。失败与冒烟记录保留在索引中，但不会进入 W2 结果。已记录的算法与测量差异见 [`docs/deviations.md`](docs/deviations.md)。

每个 run 目录都是自解释的，这正是本仓库的设计目的：

```
runs/20261030-1247_HalfCheetah-v4_seed0_clip0.2_smoke/
├─ config.json     命令行、解析后的超参、git commit、机器信息、耗时、返回码
├─ progress.csv    每次更新一行：回报、各项 loss、approx_kl、clipfrac……
├─ stdout.log      子进程的原始控制台输出
└─ tensorboard/    事件文件，已被 gitignore（二进制、diff 里不可读）
```

## 目录结构

```
src/ppo_continuous_action.py   vendored 的 CleanRL PPO + [REPRO] 日志钩子
src/ppo_args.py                runner 与训练器共享的参数、默认值与校验
scripts/run.py                 唯一跑训练的入口
scripts/run_baseline.py        W2 三环境矩阵与验收清单
scripts/verify_runs.py         检查步数、指标、配置、索引和冻结版本
scripts/plot.py                progress.csv -> PNG（图从不手画）
scripts/summarize_baseline.py  完整 W2 run -> 可追溯摘要
scripts/collect_env.py         机器与库版本快照 -> env.txt
scripts/weekly_check.ps1       推送前自查，规则与评分项一一对应
configs/base.yaml              论文 Table 3 + 实验矩阵（作为记录，不作为真源）
docs/setup.md                  装环境时真实踩过的坑
docs/deviations.md             与论文的每一处差异
runs/                          提交进仓库的证据，一个 run 一个目录
logs/runs.csv                  所有 run 的索引
```

本仓库的代码与文字在 AI 助手辅助下起草；实验数值由本机实际 MuJoCo/PyTorch 训练产生。自动执行与人工复核的边界记录在 [`AI_USAGE.md`](AI_USAGE.md)，未获人工签名的内容不会冒称已由组员复核。

# MIAE01 论文复现：MuJoCo 上的 PPO

课程项目（teamwork-PPOA）。我们从论文本身复现近端策略优化（Proximal Policy Optimization，arXiv:1707.06347v2），而不是调用框架里现成的 `PPO()`；并且检验论文关于"三种代理目标里哪一种真正有效"的核心主张，能否在我们自己的测量中成立。

> **状态：七周中的第 1 周，进行中。** 仓库骨架、vendored 训练器、运行与日志管线已经就位；实验矩阵尚未开始。本 README 里的任何内容都不构成"结果"——何时交付什么，见文末的七周路线图。英文版见 [`docs/README.en.md`](docs/README.en.md)。

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
| **W1** | 10/26 – 11/01 | 仓库地基、环境钉版本、管线跑通 | `env.txt`、一次冒烟跑的 `progress.csv` + `config.json` + `stdout.log`、第一张曲线、偏差表起头 |
| W2 | 11/02 – 11/08 | baseline 打通到全尺度 | 3 环境 × seed 0 × 100 万步；`logs/runs.csv` 索引；所有参数改成命令行开关；冻结 gymnasium / MuJoCo 版本 |
| W3 | 11/09 – 11/15 | **C1：论文的主 claim** | `clip0.2` / `adaptiveKL` / `noclip` × 3 环境 × 3 种子 = 27 次；每个变体只是一个开关，不改源码 |
| W4 | 11/16 – 11/22 | **C2** + 自动出表 | `update_epochs` 1 与 10 的对照；`scripts/make_table.py` 自动生成 Paper vs Ours 表；报告开始动笔 |
| W5 | 11/23 – 11/29 | Extension（单因子） | `K` 扫描 × 3 环境 × 3 种子，一张图，一句能站住的结论 |
| W6 | 11/30 – 12/06 | 差异归因 + 初稿 | 至少一次受控重跑作为差异证据；偏差表定稿；报告初稿齐全 |
| W7 | 12/07 – 12/12 | 定稿 | 报告 PDF、展示、`v1.0` tag；由非本组同学照 README 从零走通一次 |

## 快速开始

需要 Python 3.10；版本全部用 `==` 钉在 [`requirements.txt`](requirements.txt)，实测机器与全部依赖版本记录在 [`env.txt`](env.txt)。

```powershell
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

python scripts/run.py --env HalfCheetah-v4 --seed 0 --variant clip0.2 --smoke
python scripts/plot.py --run runs/<run_id> --out figures/w1_smoke_HalfCheetah.png
```

**预期输出与运行时长**

| 命令 | 步数 | 预期耗时 | 产出 |
|---|---|---|---|
| `run.py --smoke` | 20,000（9 次更新） | 约 1 分钟 | `runs/<id>/progress.csv`（9 行）、`config.json`、`stdout.log` |
| `run.py`（完整） | 1,000,000 | 约 55 分钟（参考机器） | 约 488 次更新，每次一行 |

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
scripts/run.py                 唯一跑训练的入口
scripts/plot.py                progress.csv -> PNG（图从不手画）
scripts/make_table.py          runs -> Paper vs Ours 表
scripts/collect_env.py         机器与库版本快照 -> env.txt
scripts/weekly_check.ps1       推送前自查，规则与评分项一一对应
configs/base.yaml              论文 Table 3 + 实验矩阵（作为记录，不作为真源）
docs/setup.md                  装环境时真实踩过的坑
docs/deviations.md             与论文的每一处差异
runs/                          提交进仓库的证据，一个 run 一个目录
logs/runs.csv                  所有 run 的索引
```

本仓库的代码与文字在 AI 助手辅助下起草；AI 不产出任何实验结果。逐文件的记录（含每位成员读过、跑过哪些文件）在 [`AI_USAGE.md`](AI_USAGE.md)。

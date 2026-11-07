# 七周路线图（操作版）

> 本文件是**按周执行的工作清单**；[`README.md`](README.md) 负责讲"这个项目是什么、为什么这么做"。
> 两者一旦冲突，以本文件的日期与交付物为准，并在改动当周同步回 README。

## 基准

| | |
|---|---|
| 今天 | 2026-11-07（周六） |
| 当前周 | **W1：11/02 – 11/08** |
| 周窗口 | 周一 00:00 – 周日 23:59（周检的统计口径就是这个窗口） |
| 每周硬性要求 | 每人至少 1 条**已推送**的实质 commit；首行含学号 + 姓名 |
| 分支策略 | 只推 `main`；本仓库规模不需要功能分支 |

日期基准：整套计划**整体顺延一周**，第 1 周从 11/02 起算。

> ⚠️ 原计划里 **12/12 是展示日**。顺延后 W7 结束在 12/20，比展示日晚 8 天。
> 若课程展示日仍固定在 12/12，则 W6 与 W7 必须在 12/07 – 12/12 内合并完成——**这条要跟课程通知核实**。

## 七周总表

| 周次 | 日期（检查点：周日 23:59） | 目标 | 交付物 |
|---|---|---|---|
| **W1** | 11/02 – 11/08 | 仓库地基、环境钉版本、管线跑通 | `env.txt`、一次冒烟跑的 `progress.csv` + `config.json` + `stdout.log`、第一张曲线、偏差表起头 |
| W2 | 11/09 – 11/15 | baseline 打通到全尺度 | 3 环境 × seed 0 × 100 万步；`logs/runs.csv` 索引；所有参数改成命令行开关；冻结 gymnasium / MuJoCo 版本 |
| W3 | 11/16 – 11/22 | C1：论文的主 claim | `clip0.2` / `adaptiveKL` / `noclip` × 3 环境 × 3 种子 = 27 次；每个变体只是一个开关，不改源码 |
| W4 | 11/23 – 11/29 | C2 + 自动出表 | `update_epochs` 1 与 10 的对照；`make_table.py` 自动生成 Paper vs Ours 表；报告开始动笔 |
| W5 | 11/30 – 12/06 | Extension（单因子） | `K` 扫描 × 3 环境 × 3 种子，一张图，一句能站住的结论 |
| W6 | 12/07 – 12/13 | 差异归因 + 初稿 | 至少一次受控重跑作为差异证据；偏差表定稿；报告初稿齐全 |
| W7 | 12/14 – 12/20 | 定稿 | 报告 PDF、展示、`v1.0` tag；由非本组同学照 README 从零走通一次 |

## W1 详细清单（本周）

### 已完成

- [x] `git init -b main`，仓库治理文件就位（`.gitignore` / `.gitattributes` / `.gitmessage` / `githooks/commit-msg` / `LICENSE`）
- [x] `requirements.txt` 全部 `==` 钉死；`configs/base.yaml` 录入论文 Table 3
- [x] vendor CleanRL PPO @ `cf20043`，附上游 MIT 许可与 `MODIFICATIONS.md`
- [x] 统一 runner `scripts/run.py` 与 `scripts/collect_env.py`
- [x] README（中/英）与本文路线图
- [x] 提交前钩子：首行缺学号直接拒绝（已用 5 个用例回归）

### 待完成（本周内）

- [ ] 跑一次冒烟：`python scripts/run.py --env HalfCheetah-v4 --seed 0 --variant clip0.2 --smoke`
- [ ] 恢复并运行 `scripts/plot.py`，产出 `figures/w1_smoke_HalfCheetah.png`
- [ ] `python scripts/collect_env.py` 生成 `env.txt` 并提交
- [ ] 恢复 `scripts/weekly_check.ps1`（推送前自查用，「每周例行」第 2 步依赖它）
- [ ] `docs/deviations.md` 写出前 3 条偏差
- [ ] `docs/setup.md` 记录真实踩坑（环境已通，补记录即可）
- [ ] `AI_USAGE.md` 签名（`<name>` / `<date>` 必须替换成真人真日期）
- [ ] README 团队表填三行（学号 / 姓名 / 分工）

### 本周不做什么

- 不跑 100 万步正式实验（W2 的事），W1 的 KPI 是"链路通 + 有日志 + 有图"
- 不动超参：W1–W5 全部用论文 Table 3 原值，任何偏离先进偏差表
- 不写报告正文（W4 起步）

## 每周例行（周日 20:00 前完成，别卡 23:59）

1. `git pull --rebase` 拉队友的改动
2. 跑一次自查：`powershell -NoProfile -File scripts\weekly_check.ps1 -Expected <学号1>,<学号2>,<学号3>`
3. 确认本周窗口内**每人**都有一条已推送的 commit
4. 确认 `git status` 干净、`git ls-files` 里没有权重/数据集/tensorboard 事件
5. `git push`，再刷新 GitHub 确认 `main` 上确实有这批提交

## 里程碑与证据链

| 里程碑 | 完成标志 | 对应评分 |
|---|---|---|
| 仓库可复现 | 陌生人照 README 从零装好并跑出第一张图 | Reproducibility 25% |
| 主 claim 有数据 | 27 次 run 的三档对照表 | Results 30% |
| 批判分析有弹药 | `clipfrac` / `approx_kl` 随训练的变化能解释清楚 | Critical Analysis 30% |
| 提交历史合规 | 七周窗口内每周每人至少一条合规 commit | Weekly Commits 30% / Individual Contributions 10% |

# 七周路线图（操作版）

> 本文件是**按周执行的工作清单**；[`README.md`](README.md) 负责讲"这个项目是什么、为什么这么做"。
> 两者一旦冲突，以本文件的日期与交付物为准，并在改动当周同步回 README。

## 基准

| | |
|---|---|
| 今天 | 2026-11-27（周五） |
| 当前周 | **W4：11/23 – 11/29** |
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
| W3 | 11/16 – 11/22 | C1：论文的主 claim | `clip0.2` / `adaptiveKL` / `noclip` × 3 环境 × 3 种子 = 27 次；`noclip` 只是换 `--clip-coef`，`adaptiveKL` 需要先实现（计划原文写的是"不改源码"，实际新增了 KL 罚项实现，见下方 W3 详细清单） |
| W4 | 11/23 – 11/29 | C2 + 自动出表 | `update_epochs` 1 与 10 的对照；`make_table.py` 自动生成 Paper vs Ours 表；报告开始动笔 |
| W5 | 11/30 – 12/06 | Extension（单因子） | `K` 扫描 × 3 环境 × 3 种子，一张图，一句能站住的结论 |
| W6 | 12/07 – 12/13 | 差异归因 + 初稿 | 至少一次受控重跑作为差异证据；偏差表定稿；报告初稿齐全 |
| W7 | 12/14 – 12/20 | 定稿 | 报告 PDF、展示、`v1.0` tag；由非本组同学照 README 从零走通一次 |

## W2 详细清单（本周）

- [x] runner 与训练器共享参数定义，超参通过命令行显式传递并记录实际解析值
- [x] 冻结 Gymnasium 0.29.1 / MuJoCo 3.1.6，生成完整依赖锁和机器快照
- [x] 运行目录唯一命名；失败状态与实际步数入索引；并发写入加锁
- [x] 补足最后一批，确保请求 1,000,000 步时实际完成 1,000,000 步（489 次更新）
- [x] 10 项回归测试通过，包含真实单/双环境短训练、末批处理和失败路径
- [x] 20,000 步冒烟通过并出图；记录中文路径和控制台编码问题
- [x] 三环境 × seed 0 × clip0.2 × 1,000,000 步全部完成，验收 `logs/w2_baseline.json`
- [x] 生成全尺度曲线和可追溯摘要，见 `docs/w2_baseline.md`
- [x] 核对锁定的 CleanRL 上游：损失/优化器、GAE 递推、包装链与默认网络初始化共 7 项通过
- [x] 推送并核对远端 main：W2 实现与证据提交 `0732e76` 已发布

执行：`python scripts/run_baseline.py --jobs 3 --threads 1`。
验收：`python scripts/verify_runs.py --baseline`。
安装与环境限制见 [`docs/setup.md`](docs/setup.md)，算法/测量口径见 [`docs/deviations.md`](docs/deviations.md)。

W2 只打通基线，不把单种子训练曲线当作 C1 排序的证据。
真正的 `adaptiveKL` 罚项已在 W3 实现（见下）。

## W3 详细清单（2026-11-22，本轮由张崇彬轮值）

- [x] 实现 KL 罚项代理目标：`--surrogate kl-penalty` + β 自适应（论文 Algorithm 2）+ β 上界开关
- [x] 回归测试从 12 项扩到 13 项并全绿；原先"`adaptiveKL` 必须被拒"的用例改为非法变体名
- [x] 主矩阵 27 次：`clip0.2` / `adaptiveKL` / `noclip` × 3 环境 × 3 种子 × 100 万步，全部 `completed`
- [x] 受控对照 18 次：`adaptiveKL-capped`（β ≤ 100）、`adaptiveKL-tight`（β ≤ 1）
- [x] `scripts/plot_c1.py` 出两张图（柱状图 + 学习曲线），数据全部读自 `progress.csv`
- [x] 结果与裁决写入 [`docs/w3_c1.md`](docs/w3_c1.md)

执行：`python scripts/run.py --variant <variant> --env <env> --seed <seed> --threads 1`
验收：`python scripts/verify_runs.py --run runs/<run_id>`；汇总与出图见 `docs/w3_c1.md` §6。

W3 的结论是**部分复现**：裁剪优于不裁剪、裁剪优于 KL 罚项这两条成立；
但三个 β 设置训练轨迹逐行相同，说明我们的 KL 罚项实现退化成了"只最小化 KL"，
**不能**用来判定论文关于中间那一档的排序。这是本轮的已知局限，写进了偏差表。

## W4 详细清单（2026-11-27，本轮由张崇彬轮值）

- [x] `scripts/run.py --label`：同一 variant 的单因子对照写进 run_id（`..._clip0.2_k1_full`），标签只进运行档案、不进训练器
- [x] `scripts/run_c2.py` 跑 18 次：K=1 / K=10 × 3 环境 × 3 种子 × 100 万步，两个臂在同一台机器、同一依赖、同一预算下产生
- [x] `scripts/verify_runs.py --c2`：两臂只允许 `update_epochs` 不同，环境/种子/线程/依赖/平台必须一致，run_id 必须编码 K
- [x] `scripts/summarize_c2.py` + `plot_c2.py`：`logs/c2_summary.json` + 两张图
- [x] `scripts/make_table.py`：Paper Table 1 vs Ours（mean±std、逐任务排名、子 claim 检查、run 溯源）
- [x] `scripts/make_all.py`：一条命令重跑全部摘要、图、表并做证据校验
- [x] 回归测试 13 → 26 项，全绿
- [x] `docs/w4_c2.md`、生成的 `docs/paper_vs_ours.md`
- [x] 报告 `setup` 与 `results` 两节起稿（`docs/report.md`）
- [x] 偏差表补到 W2 13 + W3 7 + W4 12 = 32 行

执行：`python scripts/run_c2.py --jobs 9 --threads 1`
验收：`python scripts/verify_runs.py --c2`，然后 `python scripts/make_all.py`

W4 的结论：**K = 10 是承重的默认值**。同样 1M 步预算下，K = 10 在 3/3 环境、
9/9 同种子对胜出（+38%～+250%），代价约 1.4 倍墙钟；末 50 次更新的诊断量显示
K = 1 时 `approx_kl` 只有 4–8×10⁻⁵、`clipfrac` 全为 0.000 —— 裁剪机制几乎不触发。
这把 C2 和 C1 连起来：论文的裁剪只有在 K > 1 时才有东西可裁。
局限：per-minibatch 优势归一化让 K 里同时混着"多归一化"，现有实现拆不开（W6 可补）。

W4 不碰训练代码：`update_epochs` 从 W2 起就是命令行参数，两个臂跑的是同一份
vendored trainer；runner 侧唯一改动是 `--label`。

W4 没有做的事：不跑 C3 的 K 扫描（W5 的事）；不重跑 C1（W3 的 27 + 18 次仍是 C1
证据，但要说明它们来自另一台机器，见 `docs/deviations.md` 的 W4 段）。


## W1 详细清单（历史）

### 已完成

- [x] `git init -b main`，仓库治理文件就位（`.gitignore` / `.gitattributes` / `.gitmessage` / `githooks/commit-msg` / `LICENSE`）
- [x] `requirements.txt` 全部 `==` 钉死；`configs/base.yaml` 录入论文 Table 3
- [x] vendor CleanRL PPO @ `cf20043`，附上游 MIT 许可与 `MODIFICATIONS.md`
- [x] 统一 runner `scripts/run.py` 与 `scripts/collect_env.py`
- [x] README（中/英）与本文路线图
- [x] 提交前钩子：首行缺学号直接拒绝（已用 5 个用例回归）

### 原待办（W1 未随仓库交付的证据）

- [ ] 跑一次冒烟：`python scripts/run.py --env HalfCheetah-v4 --seed 0 --variant clip0.2 --smoke`
- [ ] 恢复并运行 `scripts/plot.py`，产出 `figures/w1_smoke_HalfCheetah.png`
- [ ] `python scripts/collect_env.py` 生成 `env.txt` 并提交
- [ ] 恢复 `scripts/weekly_check.ps1`（推送前自查用，「每周例行」第 2 步依赖它）
- [ ] `docs/deviations.md` 写出前 3 条偏差
- [ ] `docs/setup.md` 记录真实踩坑（环境已通，补记录即可）
- [ ] `AI_USAGE.md` 签名（`<name>` / `<date>` 必须替换成真人真日期）
- [ ] README 团队表填三行（学号 / 姓名 / 分工）

2026-11-15 补充：冒烟、绘图、环境快照、自查脚本、安装和偏差说明已在 W2
实际补齐，不回填为 W1 按时完成。成员签名与团队信息仍须本人提供，不代填。

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

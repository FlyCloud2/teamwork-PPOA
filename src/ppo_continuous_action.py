#!/usr/bin/env python
"""连续动作空间下的 PPO —— CleanRL 的 vendored 副本。

这是什么
    CleanRL 单文件 PPO 实现的逐字副本，取自一个锁定的 commit，
    外加少量以 [REPRO] 标记的本地改动。
    我们选择 vendor，而不是 pip install 或跟随浮动的 submodule，
    好让"我们复现的那个算法"和自己的脚本一起钉进 git 历史。

    上游仓库：https://github.com/vwxyzjn/cleanrl
    Commit  ：cf20043b852deedf52dc9f90a325c6866d599005（2023-10-15）
    文件    ：cleanrl/ppo_continuous_action.py
    许可    ：MIT —— 见 LICENSES/cleanrl-LICENSE.txt
    取得时间：2026-10-30

怎么运行
    通常不直接调用本文件：scripts/run.py 负责建 run 目录、拼参数、定 CSV 路径，
    再以子进程方式把它拉起来。
    手工调用则是：

        $PY src/ppo_continuous_action.py --env-id HalfCheetah-v4 --seed 0 \
            --total-timesteps 20000 --exp-name clip0.2 \
            --csv-out runs/manual/progress.csv --tb-out runs/manual/tensorboard

    $PY 是锁定环境里的解释器（见 README 的「快速开始」）。

改了什么
    每处改动都以 [REPRO] 开头，并在 MODIFICATIONS.md 中逐条列出。
    概括起来：一条不依赖 tensorboard 的 CSV 证据链（--csv-out）、
    把 tensorboard 事件重定向进 run 目录（--tb-out）、一个模块 docstring，
    以及把所有注释与文档字符串译成中文。学习算法本身未被改动 ——
    更新计算里的任何一行都没有动。

    AI 参与：上述 [REPRO] 日志钩子由 AI 助手起草，
    随后由 <name> 于 <date> 逐行阅读并实际运行，详见 AI_USAGE.md。
"""
# 文档与实验结果见 https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
# [REPRO] +csv/pathlib：下面的 CSV 证据链要用（上游原本用不到这两个模块）。
import csv
import os
import pathlib
import random
import time
from ppo_args import parse_args  # [REPRO] Shared CLI and validation.

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter



def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # 兼容 dm_control 的 Dict 观测空间
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        # 注意：这里只传两个参数。gymnasium 还接受第三个 observation_space 参数，
        # 而它在 gymnasium 1.0 里变成了必填 —— 这正是 requirements.txt
        # 把 gymnasium 钉在 0.29.1 的原因（见 docs/setup.md 坑 #2）。
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, hidden_sizes=(64, 64), activation="tanh"):
        super().__init__()
        # [REPRO] Expose architecture without changing default initialization order.
        def network(output_size, output_std):
            layers = []
            width = int(np.prod(envs.single_observation_space.shape))
            for size in hidden_sizes:
                layers.extend([layer_init(nn.Linear(width, size)),
                               nn.Tanh() if activation == "tanh" else nn.ReLU()])
                width = size
            layers.append(layer_init(nn.Linear(width, output_size), std=output_std))
            return nn.Sequential(*layers)

        self.critic = network(1, 1.0)
        self.actor_mean = network(int(np.prod(envs.single_action_space.shape)), 0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # [REPRO] 事件文件是二进制、在 git 里不可读，所以由 runner 把它指到
    # runs/<id>/tensorboard（已被 gitignore），而不是仓库根下的 ./runs。
    writer = SummaryWriter(args.tb_out if args.tb_out else f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY：随机种子初始化
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # [REPRO] Persist the actual device in the raw evidence log.
    print(f"device={device}, torch_threads={torch.get_num_threads()}", flush=True)
    if args.cuda and device.type != "cuda":
        raise RuntimeError("--cuda true requested but CUDA is unavailable")

    # 环境构建
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.hidden_sizes, args.activation).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # 算法逻辑：轨迹存储的预分配
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY：开始与环境交互
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.num_updates

    # [REPRO] W3：自适应 KL 罚项的系数 beta。论文 Algorithm 2 把它初始化为 1，
    # 之后按实测 KL 与目标 KL 的比值每轮自动调整。
    kl_coef = args.kl_coef

    # [REPRO] CSV 证据链。每次更新一行；报告里每个数字都追溯到这份文件，
    # 而不是手填的表格。逐行写入并 flush，所以中途崩溃的 run
    # 也会把已有的证据留在磁盘上。
    csv_file, csv_writer = None, None
    if args.csv_out is not None:
        pathlib.Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.csv_out, "w", newline="")
        # lineterminator="\n"：Windows 下 csv.writer 默认写 CRLF，而 .gitattributes
        # 把本文件存成 LF —— 那样重跑一次就会显示成整文件改动。
        # 所以按仓库存储的形态输出。
        csv_writer = csv.writer(csv_file, lineterminator="\n")
        csv_writer.writerow(
            [
                "global_step", "episode_return", "episode_length",
                "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac",
                "explained_variance", "learning_rate", "wall_time_s",
            ]
        )
        csv_file.flush()
    # [REPRO] CleanRL 只在 episode 结束时「打印」统计，而 episode 很少正好落在
    # 更新边界上 —— 于是把最近一次结束 episode 的数值带下来，
    # 盖到下一条 CSV 行上。
    last_ep_ret, last_ep_len = float("nan"), float("nan")

    for update in range(1, num_updates + 1):
        # [REPRO] Consume the exact requested budget, including the final remainder.
        rollout_steps = min(args.num_steps, (args.total_timesteps - global_step) // args.num_envs)
        batch_size = rollout_steps * args.num_envs
        minibatch_size = batch_size // args.num_minibatches
        # 若启用了退火，按进度衰减学习率。
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(rollout_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # 算法逻辑：动作采样
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY：与环境交互并记录数据。
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # 只在至少有一个 env 结束时才打印
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # 跳过没有结束的 env
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                # [REPRO] 保存刚打印、并写进 tensorboard 的数值。
                # info["episode"]["r"] 是单元素数组，而 float(数组) 自 NumPy 1.25
                # 起已被弃用 —— 这里显式取出标量。
                last_ep_ret = float(np.asarray(info["episode"]["r"]).reshape(-1)[0])
                last_ep_len = float(np.asarray(info["episode"]["l"]).reshape(-1)[0])

        # episode 未结束时用价值函数做 bootstrap
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # 把 batch 摊平成一维
        b_obs = obs[:rollout_steps].reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs[:rollout_steps].reshape(-1)
        b_actions = actions[:rollout_steps].reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages[:rollout_steps].reshape(-1)
        b_returns = returns[:rollout_steps].reshape(-1)
        b_values = values[:rollout_steps].reshape(-1)

        # 优化策略网络与价值网络
        b_inds = np.arange(batch_size)
        clipfracs = []
        # [REPRO] W3：收集本轮每次 minibatch 的 KL，用于按论文公式整轮平均后调 beta。
        kl_means = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # 计算 approx_kl，方法见 http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    # [REPRO] PPO 论文式(10)：用于「目标 KL 早停」规则的 KL 估计。
                    # 之所以给它一个名字，是为了让 CSV 能带上这一列 ——
                    # W6 关于「裁剪到底有没有用」的批判要依赖它。
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # [REPRO] 式(7)：ratio 跳出信任域 [1-eps, 1+eps] 的样本比例；
                    # 这是「裁剪是否真的在起作用」最直接的度量。
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 策略损失
                if args.surrogate == "kl-penalty":
                    # [REPRO] W3 / 论文式(4)：不裁剪的目标，KL 罚项在下面加到总损失里。
                    # 这样 CSV 的 policy_loss 列在两种代理目标下都是「纯目标项」，可直接对比。
                    pg_loss = (-mb_advantages * ratio).mean()
                    kl_means.append(float(approx_kl.item()))
                else:
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 价值损失
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                if args.surrogate == "kl-penalty":
                    # [REPRO] W3：论文式(4) 的罚项 —— 目标项减去 beta·KL。
                    # 最小化 loss 等价于最大化「目标 − beta·KL」。
                    loss = loss + kl_coef * approx_kl

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # [REPRO] W3 / 论文第 4 节 Algorithm 2：按整轮实测的 KL 调整 beta。
        # d < d_targ / 1.5 说明走得太保守，beta 减半；d > d_targ * 1.5 说明走得太远，beta 加倍。
        # 调整只在下一轮生效，本轮所有 minibatch 用的是同一个 beta —— 与论文描述一致。
        if args.surrogate == "kl-penalty" and kl_means:
            measured_kl = float(np.mean(kl_means))
            previous_beta = kl_coef
            if measured_kl < args.kl_target / 1.5:
                kl_coef = kl_coef / 2.0
            elif measured_kl > args.kl_target * 1.5:
                kl_coef = kl_coef * 2.0
            # [REPRO] W3：上界在调整之后施加，beta 永远不会越过它。
            # 不设上界时（默认 1e300）这里等于没做任何事，字面行为保持不变。
            kl_coef = min(kl_coef, args.kl_coef_max)
            if kl_coef != previous_beta:
                print(
                    f"[kl-penalty] update={update} measured_kl={measured_kl:.6f} "
                    f"beta {previous_beta:.6f} -> {kl_coef:.6f}",
                    flush=True,
                )

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        # [REPRO] explained variance 用来说明 critic 确实在跟踪回报；
        # 当这一批回报是常数时（除以零）取 NaN。
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY：为画图记录奖励
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # [REPRO] 与上面同一批数字，落到 git 承载得住的地方
        if csv_writer is not None:
            csv_writer.writerow(
                [
                    global_step,
                    last_ep_ret,
                    last_ep_len,
                    pg_loss.item(),
                    v_loss.item(),
                    entropy_loss.item(),
                    approx_kl.item(),
                    float(np.mean(clipfracs)),
                    explained_var,
                    optimizer.param_groups[0]["lr"],
                    round(time.time() - start_time, 3),
                ]
            )
            csv_file.flush()

    envs.close()
    writer.close()
    # [REPRO] 关闭证据链，保证最后一行不会丢
    if csv_file is not None:
        csv_file.close()

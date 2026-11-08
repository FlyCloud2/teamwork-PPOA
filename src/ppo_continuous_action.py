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
import argparse
# [REPRO] +csv/pathlib：下面的 CSV 证据链要用（上游原本用不到这两个模块）。
import csv
import os
import pathlib
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # [REPRO] 输出目的地。两者默认都是 None，所以不传时上游行为
    # （tensorboard 写进 ./runs/<run_name>）保持不变。
    parser.add_argument("--csv-out", type=str, default=None,
        help="path of the per-update progress CSV; the only durable evidence trail")
    parser.add_argument("--tb-out", type=str, default=None,
        help="directory for tensorboard event files (default: runs/<run_name>)")

    # 算法相关参数
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="the coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


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
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
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

    # 环境构建
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

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
    num_updates = args.total_timesteps // args.batch_size

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
        # 若启用了退火，按进度衰减学习率。
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
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
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # 把 batch 摊平成一维
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 优化策略网络与价值网络
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
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

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

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

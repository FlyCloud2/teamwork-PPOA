"""Shared CLI schema for the runner and the vendored trainer (stdlib only)."""
import argparse
import math


def strtobool(value):
    value = value.lower()
    if value in ("1", "true", "yes", "on", "y", "t"):
        return True
    if value in ("0", "false", "no", "off", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean: true/false")


def build_parser():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_continuous_action",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
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
    # [REPRO] W3：论文 Table 1 比较的三种代理目标里，第二种是「自适应 KL 罚项」。
    # 这里用一个显式开关选代理目标，而不是在训练器里猜 variant 名字 ——
    # 保证 C1 的两次对照只差这一个因素。
    parser.add_argument("--surrogate", choices=["clip", "kl-penalty"], default="clip",
        help="policy surrogate objective: the clipped objective, or the KL penalty of the paper")
    parser.add_argument("--kl-coef", type=float, default=1.0,
        help="initial coefficient beta of the KL penalty (paper Algorithm 2 starts at 1)")
    parser.add_argument("--kl-target", type=float, default=0.01,
        help="target KL divergence d_targ used to adapt beta")
    # [REPRO] W3：论文 Algorithm 2 没有给 beta 设上界。字面实现下 beta 每轮翻倍，
    # 在 489 轮里能涨到 1e150，罚项压过一切、策略冻结（见 logs/c1_summary.json）。
    # 默认 1e300 等价于「不设上界」，所以字面版仍然可复现；
    # 受控对照版把上界设成 1 / d_targ，让 1% 的 KL 偏移大约值 1 个单位的惩罚。
    parser.add_argument("--kl-coef-max", type=float, default=1e300,
        help="upper bound for the adaptive beta; 1e300 means effectively unbounded")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--adam-eps", type=float, default=1e-5)
    return parser


def validate_args(args, parser):
    for key in ("total_timesteps", "num_envs", "num_steps", "num_minibatches", "update_epochs"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("learning_rate", "adam_eps", "max_grad_norm"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    for key in ("clip_coef", "ent_coef", "vf_coef"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and nonnegative")
    for key in ("gamma", "gae_lambda"):
        if not 0 <= getattr(args, key) <= 1:
            parser.error(f"--{key.replace('_', '-')} must be in [0, 1]")
    if args.target_kl is not None and (not math.isfinite(args.target_kl) or args.target_kl <= 0):
        parser.error("--target-kl must be finite and positive")
    # [REPRO] 自适应 KL 罚项的两个参数：beta 初值必须为正，目标 KL 必须为正。
    if not math.isfinite(args.kl_coef) or args.kl_coef <= 0:
        parser.error("--kl-coef must be finite and positive")
    if not math.isfinite(args.kl_target) or args.kl_target <= 0:
        parser.error("--kl-target must be finite and positive")
    if not math.isfinite(args.kl_coef_max) or args.kl_coef_max <= 0:
        parser.error("--kl-coef-max must be finite and positive")
    if args.surrogate == "kl-penalty" and args.target_kl is not None:
        parser.error("--surrogate kl-penalty cannot be combined with --target-kl (that is the early-stop rule)")
    if any(size <= 0 for size in args.hidden_sizes):
        parser.error("--hidden-sizes must be positive")
    args.batch_size = args.num_envs * args.num_steps
    if args.total_timesteps % args.num_envs:
        parser.error("--total-timesteps must be divisible by --num-envs")
    # An exact budget uses a shorter final rollout. Each minibatch must remain valid.
    sizes = [min(args.batch_size, args.total_timesteps)]
    remainder = args.total_timesteps % args.batch_size
    if remainder:
        sizes.append(remainder)
    for size in sizes:
        if size % args.num_minibatches:
            parser.error("full/final batch sizes must be divisible by --num-minibatches")
        if size // args.num_minibatches < (2 if args.norm_adv else 1):
            parser.error("minibatches need >=2 samples with advantage normalization")
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_updates = math.ceil(args.total_timesteps / args.batch_size)
    return args


def parse_args(argv=None):
    parser = build_parser()
    return validate_args(parser.parse_args(argv), parser)

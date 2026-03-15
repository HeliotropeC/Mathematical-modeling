"""
evaluate/evaluate.py
模型评估脚本：对比 CQL 策略、随机策略和启发式策略的表现。

用法:
  python -m evaluate.evaluate [--config config/config.yaml]
  python evaluate/evaluate.py
"""

import argparse
import os
import sys
from typing import Callable

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import BoilerSimulator, MixedBehaviorPolicy
from models.safe_offline_rl import build_cql_agent
from utils.safety_constraints import SafetyChecker
from utils.visualization import (
    plot_reward_comparison,
    plot_state_trajectory,
    plot_safety_heatmap,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- 策略包装器 ----------

class RandomPolicy:
    """均匀随机策略。"""

    def __init__(self, action_dim: int = 2, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.action_dim = action_dim

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=self.action_dim)


class HeuristicPolicy:
    """基于规则的启发式策略（用于基线比较）。"""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.policy = MixedBehaviorPolicy(self.rng)

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.policy._near_optimal(state)


class CQLPolicy:
    """包装 CQL 智能体为策略接口。"""

    def __init__(self, agent):
        self.agent = agent

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.agent.select_action(state, deterministic=True)


# ---------- 评估函数 ----------

def run_episode(
    env: BoilerSimulator,
    policy,
    episode_length: int = 200,
) -> dict:
    """
    运行单个 episode，收集轨迹数据。

    Returns:
        dict with keys: states, actions, rewards, total_reward, safe_steps
    """
    state = env.reset(random_init=True)
    states, actions, rewards = [state.copy()], [], []
    safe_steps = 0
    done = False

    for _ in range(episode_length):
        action = policy.act(state)
        next_state, reward, done, info = env.step(action)

        actions.append(action.copy())
        rewards.append(reward)
        states.append(next_state.copy())

        if info["safe"]:
            safe_steps += 1

        if done:
            state = env.reset(random_init=False)
        else:
            state = next_state

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "safe_steps": safe_steps,
        "safe_rate": safe_steps / episode_length,
    }


def evaluate_policy(
    policy,
    n_episodes: int = 50,
    episode_length: int = 200,
    noise_std: float = 0.02,
    seed: int = 42,
) -> dict:
    """
    多 episode 评估策略。

    Returns:
        aggregated stats
    """
    env = BoilerSimulator(noise_std=noise_std, random_seed=seed)
    checker = SafetyChecker()

    total_rewards = []
    mean_rewards = []
    safe_rates = []
    hard_viol_rates = []
    all_states = []
    last_episode = None

    for ep in range(n_episodes):
        ep_env = BoilerSimulator(noise_std=noise_std, random_seed=seed + ep)
        result = run_episode(ep_env, policy, episode_length)
        total_rewards.append(result["total_reward"])
        mean_rewards.append(result["mean_reward"])
        safe_rates.append(result["safe_rate"])

        # 硬约束违约率
        traj = result["states"]
        hard_safe = checker.batch_check_hard(traj)
        hard_viol_rates.append(1.0 - hard_safe.mean())
        all_states.append(traj)

        if ep == n_episodes - 1:
            last_episode = result

    all_states_arr = np.concatenate(all_states, axis=0)
    viol_summary = checker.evaluate_trajectory(all_states_arr)

    return {
        "total_reward_mean": float(np.mean(total_rewards)),
        "total_reward_std": float(np.std(total_rewards)),
        "mean_reward": float(np.mean(mean_rewards)),
        "safe_rate_mean": float(np.mean(safe_rates)),
        "safe_rate_std": float(np.std(safe_rates)),
        "hard_viol_rate": float(np.mean(hard_viol_rates)),
        "all_total_rewards": total_rewards,
        "trajectory_summary": viol_summary,
        "last_episode": last_episode,
    }


# ---------- 主评估流程 ----------

def evaluate(cfg: dict) -> dict:
    """
    完整评估流程：比较 CQL、随机和启发式策略。

    Returns:
        results: 各策略评估结果
    """
    eval_cfg = cfg["evaluation"]
    rl_cfg = cfg["offline_rl"]
    results_dir = eval_cfg["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    n_episodes = eval_cfg["n_episodes"]
    episode_length = eval_cfg["episode_length"]

    # ---- 加载 CQL 智能体 ----
    cql_save_path = rl_cfg["save_path"]
    agent = build_cql_agent(cfg)
    cql_pt = cql_save_path + ".pt"
    if os.path.exists(cql_pt):
        agent.load(cql_save_path)
        print(f"[Evaluate] CQL 模型加载自 '{cql_pt}'")
    else:
        print(f"[Evaluate] 警告：未找到 CQL 模型 '{cql_pt}'，使用未训练的随机初始化策略。")

    # ---- 定义策略 ----
    policies = {
        "CQL（安全离线RL）": CQLPolicy(agent),
        "启发式策略": HeuristicPolicy(seed=0),
        "随机策略": RandomPolicy(seed=0),
    }

    # ---- 评估 ----
    results = {}
    for name, policy in policies.items():
        print(f"\n[Evaluate] 评估策略: {name} ...")
        res = evaluate_policy(
            policy,
            n_episodes=n_episodes,
            episode_length=episode_length,
            noise_std=cfg["data"]["noise_std"],
            seed=cfg["data"]["random_seed"],
        )
        results[name] = res
        print(
            f"  总奖励: {res['total_reward_mean']:.2f} ± {res['total_reward_std']:.2f}"
        )
        print(f"  安全率: {res['safe_rate_mean']:.2%} ± {res['safe_rate_std']:.2%}")
        print(f"  硬约束违约率: {res['hard_viol_rate']:.2%}")

        # 安全约束详细报告
        checker = SafetyChecker()
        checker.print_report(
            results[name]["last_episode"]["states"],
            title=f"{name} 末 Episode 安全报告",
        )

    # ---- 可视化 ----
    # 奖励对比
    rewards_dict = {name: res["all_total_rewards"] for name, res in results.items()}
    plot_reward_comparison(
        rewards_dict,
        title="各策略 Episode 奖励对比",
        save_path=os.path.join(results_dir, "reward_comparison.png"),
    )

    # CQL 状态轨迹
    cql_traj = results["CQL（安全离线RL）"]["last_episode"]["states"]
    plot_state_trajectory(
        cql_traj,
        title="CQL 策略状态轨迹",
        save_path=os.path.join(results_dir, "cql_trajectory.png"),
    )
    plot_safety_heatmap(
        cql_traj,
        title="CQL 策略安全约束热图",
        save_path=os.path.join(results_dir, "cql_safety_heatmap.png"),
    )

    # 启发式轨迹
    heur_traj = results["启发式策略"]["last_episode"]["states"]
    plot_state_trajectory(
        heur_traj,
        title="启发式策略状态轨迹",
        save_path=os.path.join(results_dir, "heuristic_trajectory.png"),
    )

    # ---- 汇总 ----
    _print_summary(results)

    # 保存数值结果
    summary_lines = []
    for name, res in results.items():
        summary_lines.append(
            f"{name},"
            f"{res['total_reward_mean']:.4f},"
            f"{res['total_reward_std']:.4f},"
            f"{res['safe_rate_mean']:.4f},"
            f"{res['hard_viol_rate']:.4f}"
        )
    with open(os.path.join(results_dir, "evaluation_summary.csv"), "w") as f:
        f.write("policy,total_reward_mean,total_reward_std,safe_rate_mean,hard_viol_rate\n")
        f.write("\n".join(summary_lines))
    print(f"\n[Evaluate] 评估结果已保存至 '{results_dir}/'")

    return results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 65)
    print("  策略性能对比摘要")
    print("=" * 65)
    header = f"{'策略':<22} {'总奖励均值':>12} {'安全率':>10} {'硬约束违约率':>14}"
    print(header)
    print("-" * 65)
    for name, res in results.items():
        print(
            f"{name:<22} "
            f"{res['total_reward_mean']:>12.2f} "
            f"{res['safe_rate_mean']:>10.2%} "
            f"{res['hard_viol_rate']:>14.2%}"
        )
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="策略评估")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()

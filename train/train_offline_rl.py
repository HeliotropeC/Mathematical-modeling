"""
train/train_offline_rl.py
安全离线强化学习（CQL）训练脚本。

用法:
  python -m train.train_offline_rl [--config config/config.yaml]
  python train/train_offline_rl.py
"""

import argparse
import os
import sys

import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import (
    generate_offline_dataset,
    load_offline_dataset,
    split_dataset,
)
from models.safe_offline_rl import build_cql_agent, train_cql
from utils.visualization import plot_cql_training_stats


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_offline_rl(cfg: dict) -> list:
    """
    完整的离线 RL 训练流程。

    1. 加载 / 生成离线数据集（使用训练集部分）
    2. 构建 CQL 智能体
    3. 离线训练
    4. 保存训练统计图

    Returns:
        stats_history: 每步训练统计
    """
    data_cfg = cfg["data"]
    rl_cfg = cfg["offline_rl"]
    results_dir = cfg["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # ---- 数据 ----
    save_dir = data_cfg["save_dir"]
    if os.path.exists(os.path.join(save_dir, "states.npy")):
        print("[TrainRL] 加载已有数据集...")
        dataset = load_offline_dataset(save_dir)
    else:
        print("[TrainRL] 生成新数据集...")
        dataset = generate_offline_dataset(
            n_samples=data_cfg["n_samples"],
            episode_length=data_cfg["episode_length"],
            noise_std=data_cfg["noise_std"],
            random_seed=data_cfg["random_seed"],
            save_dir=save_dir,
        )

    # 仅使用训练集进行离线 RL 训练
    train_ds, val_ds, _ = split_dataset(
        dataset,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        random_seed=data_cfg["random_seed"],
    )
    print(f"[TrainRL] 离线训练集大小: {len(train_ds['states'])} 条")

    # ---- 智能体 ----
    agent = build_cql_agent(cfg)

    # ---- 训练 ----
    stats_history = train_cql(
        agent=agent,
        dataset=train_ds,
        max_steps=rl_cfg["max_steps"],
        batch_size=rl_cfg["batch_size"],
        eval_interval=rl_cfg["eval_interval"],
        save_interval=rl_cfg["save_interval"],
        save_path=rl_cfg["save_path"],
        warmup_steps=rl_cfg["warmup_steps"],
    )

    # ---- 可视化 ----
    plot_cql_training_stats(
        stats_history,
        save_path=os.path.join(results_dir, "cql_training_stats.png"),
    )

    # 保存统计数据
    stats_arr = np.array([
        [s["step"], s["q_loss"], s["policy_loss"], s["conservative_loss"]]
        for s in stats_history
    ])
    np.save(os.path.join(results_dir, "cql_training_stats.npy"), stats_arr)
    print(f"[TrainRL] 训练统计已保存至 '{results_dir}/'")

    return stats_history


def main():
    parser = argparse.ArgumentParser(description="安全离线强化学习训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_offline_rl(cfg)


if __name__ == "__main__":
    main()

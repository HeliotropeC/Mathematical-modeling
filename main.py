"""
main.py
燃煤锅炉运行参数安全离线优化系统 — 统一入口。

子命令:
  generate   生成离线数据集
  train-dt   训练数字孪生模型
  train-rl   训练安全离线 RL 智能体 (CQL)
  evaluate   评估并对比各策略
  all        依次执行以上所有步骤

用法示例:
  python main.py generate --config config/config.yaml
  python main.py train-dt
  python main.py train-rl
  python main.py evaluate
  python main.py all
"""

import argparse
import os
import sys
import time

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_generate(cfg: dict) -> None:
    from data.data_generator import generate_offline_dataset

    data_cfg = cfg["data"]
    generate_offline_dataset(
        n_samples=data_cfg["n_samples"],
        episode_length=data_cfg["episode_length"],
        noise_std=data_cfg["noise_std"],
        random_seed=data_cfg["random_seed"],
        save_dir=data_cfg["save_dir"],
    )


def cmd_train_dt(cfg: dict) -> None:
    from train.train_digital_twin import train_digital_twin

    train_digital_twin(cfg)


def cmd_train_rl(cfg: dict) -> None:
    from train.train_offline_rl import train_offline_rl

    train_offline_rl(cfg)


def cmd_evaluate(cfg: dict) -> None:
    from evaluate.evaluate import evaluate

    evaluate(cfg)


def cmd_all(cfg: dict) -> None:
    print("\n" + "=" * 60)
    print("  燃煤锅炉运行参数安全离线优化系统")
    print("  完整流程运行")
    print("=" * 60)

    steps = [
        ("生成数据集", cmd_generate),
        ("训练数字孪生", cmd_train_dt),
        ("训练离线 RL (CQL)", cmd_train_rl),
        ("策略评估", cmd_evaluate),
    ]

    for name, func in steps:
        print(f"\n>>> {name} ...")
        t0 = time.time()
        func(cfg)
        elapsed = time.time() - t0
        print(f"<<< {name} 完成，耗时 {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("  全部流程完成！结果保存在 results/ 目录")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="燃煤锅炉运行参数安全离线优化系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=["generate", "train-dt", "train-rl", "evaluate", "all"],
        help="执行的子命令",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径 (默认: config/config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    dispatch = {
        "generate": cmd_generate,
        "train-dt": cmd_train_dt,
        "train-rl": cmd_train_rl,
        "evaluate": cmd_evaluate,
        "all": cmd_all,
    }
    dispatch[args.command](cfg)


if __name__ == "__main__":
    main()

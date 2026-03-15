"""
train/train_digital_twin.py
数字孪生模型训练脚本。

用法:
  python -m train.train_digital_twin [--config config/config.yaml]
  python train/train_digital_twin.py
"""

import argparse
import os
import sys

import numpy as np
import yaml
from torch.utils.data import DataLoader

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import (
    generate_offline_dataset,
    load_offline_dataset,
    split_dataset,
)
from models.digital_twin import (
    BoilerSequenceDataset,
    DigitalTwinTrainer,
    LSTMDigitalTwin,
    build_digital_twin,
)
from utils.visualization import plot_training_curves, plot_digital_twin_prediction


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_digital_twin(cfg: dict) -> dict:
    """
    完整的数字孪生训练流程。

    1. 生成 / 加载离线数据集
    2. 构建序列数据集和 DataLoader
    3. 初始化模型和训练器
    4. 训练 & 早停
    5. 保存曲线图

    Returns:
        history: {"train_loss": [...], "val_loss": [...]}
    """
    data_cfg = cfg["data"]
    dt_cfg = cfg["digital_twin"]
    results_dir = cfg["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # ---- 数据 ----
    save_dir = data_cfg["save_dir"]
    if os.path.exists(os.path.join(save_dir, "states.npy")):
        print("[TrainDT] 加载已有数据集...")
        dataset = load_offline_dataset(save_dir)
    else:
        print("[TrainDT] 生成新数据集...")
        dataset = generate_offline_dataset(
            n_samples=data_cfg["n_samples"],
            episode_length=data_cfg["episode_length"],
            noise_std=data_cfg["noise_std"],
            random_seed=data_cfg["random_seed"],
            save_dir=save_dir,
        )

    train_ds, val_ds, test_ds = split_dataset(
        dataset,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        random_seed=data_cfg["random_seed"],
    )

    seq_len = dt_cfg["seq_len"]
    batch_size = dt_cfg["batch_size"]

    train_seq = BoilerSequenceDataset(
        train_ds["states"], train_ds["actions"], train_ds["next_states"], seq_len
    )
    val_seq = BoilerSequenceDataset(
        val_ds["states"], val_ds["actions"], val_ds["next_states"], seq_len
    )
    test_seq = BoilerSequenceDataset(
        test_ds["states"], test_ds["actions"], test_ds["next_states"], seq_len
    )

    train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_seq, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_seq, batch_size=batch_size, shuffle=False)

    # ---- 模型 ----
    model = build_digital_twin(cfg)
    trainer = DigitalTwinTrainer(
        model,
        lr=dt_cfg["learning_rate"],
        grad_clip=dt_cfg["grad_clip"],
    )

    save_path = dt_cfg["save_path"]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # ---- 训练 ----
    history = trainer.fit(
        train_loader,
        val_loader,
        max_epochs=dt_cfg["max_epochs"],
        patience=dt_cfg["patience"],
        save_path=save_path,
    )

    # ---- 测试集评估 ----
    test_loss = trainer.validate(test_loader)
    print(f"[TrainDT] 测试集 MSE Loss: {test_loss:.6f}")

    # ---- 可视化 ----
    plot_training_curves(
        history["train_loss"],
        history["val_loss"],
        title="数字孪生训练/验证损失",
        save_path=os.path.join(results_dir, "dt_training_curve.png"),
    )

    # 预测 vs 真实对比
    import torch

    model.eval()
    x_sample, y_sample = test_seq[:100]
    with torch.no_grad():
        y_pred = model(x_sample).numpy()

    from data.data_generator import STATE_MIN, STATE_RANGE

    true_states = y_sample.numpy() * STATE_RANGE + STATE_MIN
    pred_states = y_pred * STATE_RANGE + STATE_MIN
    plot_digital_twin_prediction(
        true_states,
        pred_states,
        n_steps=100,
        save_path=os.path.join(results_dir, "dt_prediction.png"),
    )

    return history


def main():
    parser = argparse.ArgumentParser(description="数字孪生模型训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_digital_twin(cfg)


if __name__ == "__main__":
    main()

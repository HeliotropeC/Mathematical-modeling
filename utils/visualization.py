"""
utils/visualization.py
训练过程可视化与结果分析图表。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示器环境
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data.data_generator import STATE_NAMES, STATE_MIN, STATE_MAX


plt.rcParams.update({
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    title: str = "数字孪生训练曲线",
    save_path: str = None,
) -> None:
    """绘制数字孪生训练/验证损失曲线。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="训练损失", color="steelblue")
    ax.plot(epochs, val_losses, label="验证损失", color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        print(f"[Viz] 训练曲线保存至 '{save_path}'")
    plt.close()


def plot_cql_training_stats(
    stats_history: list,
    save_path: str = None,
) -> None:
    """绘制 CQL 训练统计（Q Loss, Policy Loss, Conservative Loss）。"""
    steps = [s["step"] for s in stats_history]
    q_loss = [s["q_loss"] for s in stats_history]
    p_loss = [s["policy_loss"] for s in stats_history]
    cons_loss = [s["conservative_loss"] for s in stats_history]

    # 平滑
    def smooth(arr, k=100):
        if len(arr) < k:
            return arr
        return np.convolve(arr, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, label, color in zip(
        axes,
        [q_loss, p_loss, cons_loss],
        ["Q Loss", "Policy Loss", "Conservative Loss"],
        ["steelblue", "darkorange", "green"],
    ):
        smoothed = smooth(data)
        ax.plot(steps[: len(smoothed)], smoothed, color=color, alpha=0.9)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(label)
        ax.set_title(label)
    plt.suptitle("CQL 离线强化学习训练统计", fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        print(f"[Viz] CQL 训练统计保存至 '{save_path}'")
    plt.close()


def plot_state_trajectory(
    states: np.ndarray,
    title: str = "状态轨迹",
    save_path: str = None,
    mark_violations: bool = True,
) -> None:
    """
    绘制锅炉状态时序图（8 维分面）。

    Args:
        states: (T, 8)
        mark_violations: 是否标记超出约束的时间步
    """
    T = len(states)
    t = np.arange(T)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(4, 2, figure=fig)

    units = ["°C", "MPa", "°C", "%", "°C", "t/h", "m³/h", "mg/m³"]

    for i, (name, unit) in enumerate(zip(STATE_NAMES, units)):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.plot(t, states[:, i], color="steelblue", linewidth=0.8, label=name)
        # 安全边界
        ax.axhline(STATE_MIN[i], color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="硬下限")
        ax.axhline(STATE_MAX[i], color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="硬上限")
        if mark_violations:
            viol = (states[:, i] < STATE_MIN[i]) | (states[:, i] > STATE_MAX[i])
            if viol.any():
                ax.scatter(t[viol], states[viol, i], color="red", s=10, zorder=5)
        ax.set_ylabel(f"{name}\n({unit})", fontsize=8)
        ax.set_xlabel("Step", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        print(f"[Viz] 状态轨迹保存至 '{save_path}'")
    plt.close()


def plot_reward_comparison(
    rewards_dict: dict,
    title: str = "策略奖励对比",
    save_path: str = None,
) -> None:
    """
    对比多个策略的 episode 奖励分布。

    Args:
        rewards_dict: {策略名: [episode_reward, ...]}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = list(rewards_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    # 折线图（episode 维度）
    for (name, rewards), color in zip(rewards_dict.items(), colors):
        axes[0].plot(rewards, label=name, color=color, alpha=0.8)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("累计奖励")
    axes[0].set_title("Episode 奖励曲线")
    axes[0].legend()

    # 箱线图
    data = [rewards_dict[n] for n in names]
    bp = axes[1].boxplot(data, labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("累计奖励")
    axes[1].set_title("奖励分布箱线图")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        print(f"[Viz] 奖励对比图保存至 '{save_path}'")
    plt.close()


def plot_digital_twin_prediction(
    true_states: np.ndarray,
    pred_states: np.ndarray,
    n_steps: int = 50,
    save_path: str = None,
) -> None:
    """
    对比数字孪生预测值与真实值（选前 n_steps 步）。

    Args:
        true_states: (T, 8)
        pred_states: (T, 8)
    """
    T = min(n_steps, len(true_states))
    t = np.arange(T)
    units = ["°C", "MPa", "°C", "%", "°C", "t/h", "m³/h", "mg/m³"]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(4, 2)
    for i, (name, unit) in enumerate(zip(STATE_NAMES, units)):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.plot(t, true_states[:T, i], label="真实值", color="steelblue")
        ax.plot(t, pred_states[:T, i], label="预测值", color="darkorange", linestyle="--")
        ax.set_ylabel(f"{name} ({unit})", fontsize=8)
        ax.set_xlabel("Step", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("数字孪生预测 vs 真实", fontsize=13)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        print(f"[Viz] 预测对比图保存至 '{save_path}'")
    plt.close()


def plot_safety_heatmap(
    states: np.ndarray,
    title: str = "安全约束热图",
    save_path: str = None,
) -> None:
    """
    绘制各状态维度的归一化分布热图，标注安全区间。

    Args:
        states: (T, 8)
    """
    norm = (states - STATE_MIN[None, :]) / (STATE_MAX - STATE_MIN)[None, :]
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(norm.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(8))
    ax.set_yticklabels(STATE_NAMES, fontsize=9)
    ax.set_xlabel("时间步")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="归一化值 (0=下限, 1=上限)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        print(f"[Viz] 安全热图保存至 '{save_path}'")
    plt.close()

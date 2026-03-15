"""
models/safe_offline_rl.py
安全离线强化学习智能体：基于 Conservative Q-Learning (CQL)。

算法核心：
  - 使用保守正则项惩罚数据集外动作的 Q 值，避免分布偏移
  - 添加安全层（Safety Layer）对输出动作进行约束投影
  - 支持连续动作空间（Actor-Critic 架构）

参考文献：
  Kumar et al. (2020). Conservative Q-Learning for Offline RL. NeurIPS 2020.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from data.data_generator import STATE_MIN, STATE_MAX, STATE_RANGE


# ---------- 工具函数 ----------

def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_sizes: list,
    activation: nn.Module = nn.ReLU,
    output_activation: nn.Module = None,
) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def normalize_state(state: np.ndarray) -> np.ndarray:
    return (state - STATE_MIN) / STATE_RANGE


def denormalize_state(state_norm: np.ndarray) -> np.ndarray:
    return state_norm * STATE_RANGE + STATE_MIN


# ---------- 网络 ----------

class QNetwork(nn.Module):
    """
    双 Q 网络（Clipped Double Q-learning）。
    输入: [state_norm, action], 输出: Q 值标量。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list = None,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 256]
        self.q1 = build_mlp(state_dim + action_dim, 1, hidden_sizes)
        self.q2 = build_mlp(state_dim + action_dim, 1, hidden_sizes)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class PolicyNetwork(nn.Module):
    """
    确定性策略网络（Actor）。
    输入: state_norm, 输出: action ∈ [-1, 1]^action_dim。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list = None,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 256]
        self.net = build_mlp(
            state_dim, action_dim, hidden_sizes, output_activation=nn.Tanh
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ---------- 安全层 ----------

class SafetyLayer:
    """
    安全约束投影层。

    将动作投影到不违反状态安全约束的可行域。
    使用线性化近似：在当前状态下，约束状态转移的一阶近似边界。
    对于简单边界约束，直接截断动作增量不超过余量。
    """

    def __init__(
        self,
        action_scale: np.ndarray = None,
        safety_margin: float = 0.05,
    ):
        self.action_scale = action_scale if action_scale is not None else np.array([2.0, 20.0])
        self.safety_margin = safety_margin
        # 安全边界（考虑 margin）
        self.lo = STATE_MIN + safety_margin * STATE_RANGE
        self.hi = STATE_MAX - safety_margin * STATE_RANGE

    def project(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        将动作投影使得单步转移后状态大概率在安全域内。
        采用简化策略：当状态接近边界时缩减对应方向的动作幅度。

        Args:
            state: 当前状态（原始尺度）
            action: 归一化动作 [-1, 1]^2
        Returns:
            safe_action: 投影后动作
        """
        action = action.copy()

        # 给煤量相关 (动作 0 → 状态 5)
        F_c = state[5]
        da_coal = action[0] * self.action_scale[0]
        F_c_new = F_c + da_coal
        if F_c_new < self.lo[5]:
            da_coal = max(da_coal, self.lo[5] - F_c)
        if F_c_new > self.hi[5]:
            da_coal = min(da_coal, self.hi[5] - F_c)
        action[0] = np.clip(da_coal / self.action_scale[0], -1.0, 1.0)

        # 送风量相关 (动作 1 → 状态 6)
        F_a = state[6]
        da_air = action[1] * self.action_scale[1]
        F_a_new = F_a + da_air
        if F_a_new < self.lo[6]:
            da_air = max(da_air, self.lo[6] - F_a)
        if F_a_new > self.hi[6]:
            da_air = min(da_air, self.hi[6] - F_a)
        action[1] = np.clip(da_air / self.action_scale[1], -1.0, 1.0)

        return action

    def project_batch(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """批量投影（torch 版本，近似处理）。"""
        actions = actions.clone()
        lo = torch.tensor(self.lo, dtype=torch.float32, device=states.device)
        hi = torch.tensor(self.hi, dtype=torch.float32, device=states.device)
        scale = torch.tensor(self.action_scale, dtype=torch.float32, device=states.device)

        # 给煤量 (状态索引 5)
        F_c = states[:, 5]
        da_coal = actions[:, 0] * scale[0]
        da_coal = torch.clamp(da_coal, lo[5] - F_c, hi[5] - F_c)
        actions[:, 0] = torch.clamp(da_coal / scale[0], -1.0, 1.0)

        # 送风量 (状态索引 6)
        F_a = states[:, 6]
        da_air = actions[:, 1] * scale[1]
        da_air = torch.clamp(da_air, lo[6] - F_a, hi[6] - F_a)
        actions[:, 1] = torch.clamp(da_air / scale[1], -1.0, 1.0)

        return actions


# ---------- CQL 智能体 ----------

class CQLAgent:
    """
    Conservative Q-Learning 安全离线强化学习智能体。

    核心优化目标：
      L_CQL(Q) = E_{(s,a,r,s')~D}[
                   α * (log Σ_a' exp(Q(s,a')) - Q(s,a))   ← 保守项
                   + (Q(s,a) - (r + γ * min_Q'(s',π(s')))) ² / 2  ← Bellman 误差
                 ]
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 2,
        hidden_sizes: list = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        cql_alpha: float = 1.0,
        cql_n_actions: int = 10,
        use_safety_layer: bool = True,
        safety_penalty_coef: float = 5.0,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.cql_n_actions = cql_n_actions
        self.use_safety_layer = use_safety_layer
        self.safety_penalty_coef = safety_penalty_coef
        self.device = torch.device(device)

        hidden_sizes = hidden_sizes or [256, 256]

        # 网络
        self.q_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.q_target = copy.deepcopy(self.q_net).to(self.device)
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.policy_target = copy.deepcopy(self.policy).to(self.device)

        # 冻结目标网络
        for p in self.q_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

        # 优化器
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # 安全层
        self.safety_layer = SafetyLayer()

        # 训练统计
        self.training_stats: list = []
        self._total_steps = 0

    # ------------------------------------------------------------------
    # 目标网络软更新
    # ------------------------------------------------------------------

    def _soft_update(self, net: nn.Module, target: nn.Module) -> None:
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # CQL 损失
    # ------------------------------------------------------------------

    def _cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple:
        batch_size = states.shape[0]

        # ---- Bellman 目标 ----
        with torch.no_grad():
            next_actions = self.policy_target(next_states)
            if self.use_safety_layer:
                next_actions = self.safety_layer.project_batch(
                    next_states * torch.tensor(STATE_RANGE, device=self.device, dtype=torch.float32)
                    + torch.tensor(STATE_MIN, device=self.device, dtype=torch.float32),
                    next_actions,
                )
            q1_tgt, q2_tgt = self.q_target(next_states, next_actions)
            q_tgt = torch.min(q1_tgt, q2_tgt)
            bellman_target = rewards + self.gamma * (1.0 - dones) * q_tgt

        # ---- Bellman 误差 ----
        q1, q2 = self.q_net(states, actions)
        bellman_loss = F.mse_loss(q1, bellman_target) + F.mse_loss(q2, bellman_target)

        # ---- CQL 保守项 ----
        # 随机采样动作
        rand_actions = torch.FloatTensor(
            batch_size * self.cql_n_actions, self.action_dim
        ).uniform_(-1.0, 1.0).to(self.device)

        # 重复状态以匹配随机动作批量
        states_rep = states.unsqueeze(1).repeat(1, self.cql_n_actions, 1).view(
            batch_size * self.cql_n_actions, self.state_dim
        )
        q1_rand, q2_rand = self.q_net(states_rep, rand_actions)
        q1_rand = q1_rand.view(batch_size, self.cql_n_actions)
        q2_rand = q2_rand.view(batch_size, self.cql_n_actions)

        # 策略动作
        with torch.no_grad():
            policy_actions = self.policy(states_rep)
        q1_policy, q2_policy = self.q_net(states_rep, policy_actions)
        q1_policy = q1_policy.view(batch_size, self.cql_n_actions)
        q2_policy = q2_policy.view(batch_size, self.cql_n_actions)

        # log-sum-exp（估计 log E[exp(Q)]）
        cat_q1 = torch.cat([q1_rand, q1_policy, q1.detach()], dim=1)
        cat_q2 = torch.cat([q2_rand, q2_policy, q2.detach()], dim=1)
        conservative_loss = (
            torch.logsumexp(cat_q1, dim=1).mean()
            + torch.logsumexp(cat_q2, dim=1).mean()
            - q1.mean() - q2.mean()
        ) * self.cql_alpha

        total_q_loss = bellman_loss + conservative_loss
        return total_q_loss, bellman_loss, conservative_loss

    # ------------------------------------------------------------------
    # 策略损失
    # ------------------------------------------------------------------

    def _policy_loss(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        actions = self.policy(states)
        if self.use_safety_layer:
            states_raw = (
                states * torch.tensor(STATE_RANGE, device=self.device, dtype=torch.float32)
                + torch.tensor(STATE_MIN, device=self.device, dtype=torch.float32)
            )
            actions = self.safety_layer.project_batch(states_raw, actions)

        q_val = self.q_net.q_min(states, actions)

        # 安全正则化：状态接近边界时额外惩罚
        if self.use_safety_layer:
            states_raw = (
                states * torch.tensor(STATE_RANGE, device=self.device, dtype=torch.float32)
                + torch.tensor(STATE_MIN, device=self.device, dtype=torch.float32)
            )
            lo = torch.tensor(STATE_MIN, device=self.device, dtype=torch.float32)
            hi = torch.tensor(STATE_MAX, device=self.device, dtype=torch.float32)
            margin = 0.05 * torch.tensor(STATE_RANGE, device=self.device, dtype=torch.float32)
            viol_low = F.relu(lo + margin - states_raw).sum(dim=-1, keepdim=True)
            viol_high = F.relu(states_raw - hi + margin).sum(dim=-1, keepdim=True)
            safety_penalty = self.safety_penalty_coef * (viol_low + viol_high)
            return -(q_val - safety_penalty).mean()

        return -q_val.mean()

    # ------------------------------------------------------------------
    # 单步更新
    # ------------------------------------------------------------------

    def update(
        self,
        batch: dict,
        update_policy: bool = True,
    ) -> dict:
        """
        执行一步 CQL 更新。

        Args:
            batch: dict with keys states/actions/rewards/next_states/dones (numpy)
            update_policy: 是否同步更新策略网络
        Returns:
            stats: 损失字典
        """
        self._total_steps += 1

        def to_tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        states_raw = to_tensor(batch["states"])
        next_raw = to_tensor(batch["next_states"])
        rewards = to_tensor(batch["rewards"])
        actions = to_tensor(batch["actions"])
        dones = to_tensor(batch["dones"])

        # 归一化状态
        _min = torch.tensor(STATE_MIN, dtype=torch.float32, device=self.device)
        _range = torch.tensor(STATE_RANGE, dtype=torch.float32, device=self.device)
        states = (states_raw - _min) / _range
        next_states = (next_raw - _min) / _range

        # ---- Q 网络更新 ----
        total_q_loss, bellman_loss, cons_loss = self._cql_loss(
            states, actions, rewards, next_states, dones
        )
        self.q_optimizer.zero_grad()
        total_q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.q_optimizer.step()

        # ---- 策略更新 ----
        policy_loss_val = torch.tensor(0.0)
        if update_policy:
            policy_loss = self._policy_loss(states)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()
            policy_loss_val = policy_loss.item()

        # ---- 目标网络软更新 ----
        self._soft_update(self.q_net, self.q_target)
        self._soft_update(self.policy, self.policy_target)

        stats = {
            "q_loss": total_q_loss.item(),
            "bellman_loss": bellman_loss.item(),
            "conservative_loss": cons_loss.item(),
            "policy_loss": policy_loss_val if isinstance(policy_loss_val, float) else policy_loss_val.item(),
        }
        self.training_stats.append(stats)
        return stats

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        根据当前状态选择动作。

        Args:
            state: 原始尺度状态 (state_dim,)
            deterministic: True → 策略网络，False → 加探索噪声（不用于离线 RL）
        Returns:
            action: 归一化动作 (action_dim,)
        """
        self.policy.eval()
        state_norm = (state - STATE_MIN) / STATE_RANGE
        state_t = torch.tensor(state_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = self.policy(state_t)
        action = action_t.cpu().numpy()[0]

        if self.use_safety_layer:
            action = self.safety_layer.project(state, action)

        return np.clip(action, -1.0, 1.0)

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "q_target": self.q_target.state_dict(),
                "policy": self.policy.state_dict(),
                "policy_target": self.policy_target.state_dict(),
                "total_steps": self._total_steps,
            },
            path + ".pt",
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path + ".pt", map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.policy.load_state_dict(ckpt["policy"])
        self.policy_target.load_state_dict(ckpt["policy_target"])
        self._total_steps = ckpt.get("total_steps", 0)


# ---------- 离线训练循环 ----------

class OfflineReplayBuffer:
    """包装离线数据集，支持随机批量采样。"""

    def __init__(self, dataset: dict, device: str = "cpu"):
        self.size = len(dataset["states"])
        self.states = dataset["states"].astype(np.float32)
        self.actions = dataset["actions"].astype(np.float32)
        self.rewards = dataset["rewards"].astype(np.float32)
        self.next_states = dataset["next_states"].astype(np.float32)
        self.dones = dataset["dones"].astype(np.float32)
        self.rng = np.random.default_rng(0)

    def sample(self, batch_size: int) -> dict:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }


def train_cql(
    agent: "CQLAgent",
    dataset: dict,
    max_steps: int = 100000,
    batch_size: int = 256,
    eval_interval: int = 5000,
    save_interval: int = 10000,
    save_path: str = "checkpoints/cql_agent",
    warmup_steps: int = 1000,
) -> list:
    """
    CQL 离线训练主循环。

    Returns:
        stats_history: 每步训练统计
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    buffer = OfflineReplayBuffer(dataset)
    stats_history = []

    print(f"[CQL] 开始离线训练，共 {max_steps} 步，批量大小 {batch_size}")
    for step in range(1, max_steps + 1):
        batch = buffer.sample(batch_size)
        # 前 warmup_steps 步只更新 Q，不更新策略
        update_policy = step > warmup_steps
        stats = agent.update(batch, update_policy=update_policy)
        stats["step"] = step
        stats_history.append(stats)

        if step % eval_interval == 0:
            recent = stats_history[-eval_interval:]
            avg_q = np.mean([s["q_loss"] for s in recent])
            avg_p = np.mean([s["policy_loss"] for s in recent])
            print(
                f"[CQL] Step {step:6d}/{max_steps} | "
                f"Q Loss: {avg_q:.4f} | Policy Loss: {avg_p:.4f}"
            )

        if step % save_interval == 0:
            agent.save(save_path)
            print(f"[CQL] 检查点保存至 '{save_path}.pt'")

    agent.save(save_path)
    print(f"[CQL] 训练完成，最终模型保存至 '{save_path}.pt'")
    return stats_history


def build_cql_agent(cfg: dict) -> CQLAgent:
    """根据配置字典构建 CQL 智能体。"""
    rl_cfg = cfg.get("offline_rl", {})
    boiler_cfg = cfg.get("boiler", {})
    return CQLAgent(
        state_dim=boiler_cfg.get("state_dim", 8),
        action_dim=boiler_cfg.get("action_dim", 2),
        hidden_sizes=rl_cfg.get("hidden_sizes", [256, 256]),
        lr=rl_cfg.get("learning_rate", 3e-4),
        gamma=rl_cfg.get("gamma", 0.99),
        tau=rl_cfg.get("tau", 0.005),
        cql_alpha=rl_cfg.get("cql_alpha", 1.0),
        cql_n_actions=rl_cfg.get("cql_n_actions", 10),
        use_safety_layer=rl_cfg.get("use_safety_layer", True),
        safety_penalty_coef=rl_cfg.get("safety_penalty_coef", 5.0),
    )


if __name__ == "__main__":
    from data.data_generator import generate_offline_dataset

    ds = generate_offline_dataset(n_samples=500, save_dir="/tmp/cql_test")
    agent = CQLAgent(cql_n_actions=4)
    stats = train_cql(agent, ds, max_steps=100, batch_size=32, eval_interval=50,
                      save_interval=100, save_path="/tmp/cql_test/cql_agent")
    print(f"CQL 训练测试通过，最终 Q loss: {stats[-1]['q_loss']:.4f}")

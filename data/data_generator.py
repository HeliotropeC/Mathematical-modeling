"""
data/data_generator.py
燃煤锅炉仿真环境与离线数据集生成器。

物理仿真采用离散时间一阶差分方程近似锅炉动态：
  x_{t+1} = f(x_t, a_t) + noise

状态空间 (8 维):
  0  炉膛温度     T_f   (°C)    [900, 1200]
  1  蒸汽压力     P_s   (MPa)   [8,   14  ]
  2  蒸汽温度     T_s   (°C)    [500, 560 ]
  3  氧含量       O2    (%)     [2,   6   ]
  4  排烟温度     T_fg  (°C)    [120, 200 ]
  5  给煤量       F_c   (t/h)   [20,  40  ]
  6  送风量       F_a   (m³/h)  [200, 400 ]
  7  NOx 排放     NOx   (mg/m³) [100, 400 ]

动作空间 (2 维, 归一化到 [-1, 1]):
  0  给煤量增量   Δf_c   → 实际 ±2 t/h
  1  送风量增量   Δf_a   → 实际 ±20 m³/h
"""

import os
import numpy as np
import pandas as pd


# ---------- 物理常数 & 边界 ----------

STATE_NAMES = [
    "furnace_temp",   # 炉膛温度
    "steam_pressure", # 蒸汽压力
    "steam_temp",     # 蒸汽温度
    "o2_percent",     # 氧含量
    "flue_temp",      # 排烟温度
    "coal_feed",      # 给煤量
    "air_flow",       # 送风量
    "nox",            # NOx 排放
]

ACTION_NAMES = ["delta_coal_feed", "delta_air_flow"]

STATE_MIN = np.array([900.0, 8.0, 500.0, 2.0, 120.0, 20.0, 200.0, 100.0])
STATE_MAX = np.array([1200.0, 14.0, 560.0, 6.0, 200.0, 40.0, 400.0, 400.0])
STATE_RANGE = STATE_MAX - STATE_MIN

ACTION_SCALE = np.array([2.0, 20.0])  # 归一化动作 → 实际增量

# 报警阈值（比硬约束更严）
ALARM_MIN = np.array([920.0, 8.5, 505.0, 2.5, 125.0, 22.0, 220.0, 120.0])
ALARM_MAX = np.array([1180.0, 13.5, 555.0, 5.5, 190.0, 38.0, 380.0, 380.0])

# 最优运行区间（高效稳定）
OPTIMAL_MIN = np.array([1050.0, 10.0, 530.0, 3.0, 135.0, 28.0, 280.0, 150.0])
OPTIMAL_MAX = np.array([1150.0, 12.0, 550.0, 4.5, 160.0, 36.0, 360.0, 280.0])


class BoilerSimulator:
    """
    燃煤锅炉离散时间仿真环境。

    使用简化的物理模型描述锅炉动态，支持：
    - step()：推进一步仿真
    - reset()：重置到初始状态
    - is_safe()：检查当前状态是否安全
    - compute_reward()：计算即时奖励
    """

    def __init__(
        self,
        noise_std: float = 0.02,
        random_seed: int = 42,
        dt: float = 1.0,
    ):
        self.noise_std = noise_std
        self.dt = dt
        self.rng = np.random.default_rng(random_seed)
        self.state: np.ndarray = self._nominal_state()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _nominal_state(self) -> np.ndarray:
        """返回额定工况状态（各维度中点偏向高效区间）。"""
        return np.array([
            1100.0,  # 炉膛温度
            11.0,    # 蒸汽压力
            540.0,   # 蒸汽温度
            3.5,     # 氧含量
            145.0,   # 排烟温度
            32.0,    # 给煤量
            320.0,   # 送风量
            220.0,   # NOx
        ])

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """将状态归一化到 [0, 1]。"""
        return (state - STATE_MIN) / STATE_RANGE

    def _transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        近似物理转移函数：x_{t+1} = x_t + Δx(x_t, a_t)。

        主要关系：
        - 给煤量增加 → 炉膛温度上升、蒸汽压力/温度上升、NOx 上升
        - 送风量增加 → 氧含量上升（先升后降）、炉膛温度下降、排烟温度上升
        - 适当过量空气系数（O2 3~4%）使燃烧最充分
        """
        T_f, P_s, T_s, O2, T_fg, F_c, F_a, NOx = state
        da_coal, da_air = action * ACTION_SCALE  # 实际增量

        # 过量空气系数影响
        air_fuel_ratio = F_a / (F_c + 1e-6)  # kg_air / kg_coal (简化)
        optimal_ratio = 10.0  # 理论最优风煤比
        combustion_eff = np.clip(
            1.0 - 0.05 * abs(air_fuel_ratio - optimal_ratio) / optimal_ratio,
            0.8, 1.0,
        )

        # 炉膛温度 (受给煤量和燃烧效率影响)
        dT_f = (
            0.8 * da_coal * combustion_eff
            - 0.3 * da_air / 20.0
            + 0.05 * (1100.0 - T_f)
        ) * self.dt
        T_f_new = T_f + dT_f

        # 蒸汽压力 (正比于炉膛温度)
        dP_s = (
            0.006 * dT_f
            + 0.02 * (11.0 - P_s)
        ) * self.dt
        P_s_new = P_s + dP_s

        # 蒸汽温度 (受蒸汽压力和炉温影响)
        dT_s = (
            0.4 * dT_f
            + 0.1 * (P_s_new - P_s) * 5.0
            + 0.01 * (540.0 - T_s)
        ) * self.dt
        T_s_new = T_s + dT_s

        # 氧含量 (送风量增加→O2增加，给煤量增加→O2降低)
        dO2 = (
            0.008 * da_air
            - 0.05 * da_coal
            + 0.02 * (3.5 - O2)
        ) * self.dt
        O2_new = O2 + dO2

        # 排烟温度 (送风量增大→排烟温升高，炉温升高→排烟升高)
        dT_fg = (
            0.05 * da_air / 20.0
            + 0.02 * dT_f
            + 0.01 * (145.0 - T_fg)
        ) * self.dt
        T_fg_new = T_fg + dT_fg

        # 给煤量 & 送风量 (执行器直接控制)
        F_c_new = F_c + da_coal
        F_a_new = F_a + da_air

        # NOx (高温、高氧含量时升高)
        dNOx = (
            0.2 * max(0.0, T_f_new - 1100.0) * 0.01
            + 0.5 * max(0.0, O2_new - 4.0)
            - 0.1 * (NOx - 220.0) * 0.01
        ) * self.dt
        NOx_new = NOx + dNOx

        new_state = np.array([
            T_f_new, P_s_new, T_s_new, O2_new,
            T_fg_new, F_c_new, F_a_new, NOx_new
        ])
        return new_state

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def reset(self, random_init: bool = True) -> np.ndarray:
        """重置环境状态。"""
        if random_init:
            # 在额定值附近随机初始化
            noise = self.rng.uniform(-0.05, 0.05, size=8)
            self.state = self._nominal_state() * (1.0 + noise)
        else:
            self.state = self._nominal_state()
        self.state = np.clip(self.state, STATE_MIN, STATE_MAX)
        return self.state.copy()

    def step(self, action: np.ndarray) -> tuple:
        """
        执行一步仿真。

        Args:
            action: 归一化动作向量 [-1, 1]^2

        Returns:
            next_state, reward, done, info
        """
        action = np.clip(action, -1.0, 1.0)
        next_state = self._transition(self.state, action)

        # 加观测噪声
        noise = self.rng.normal(0.0, self.noise_std, size=8) * STATE_RANGE
        next_state = next_state + noise

        # 硬性约束截断
        next_state = np.clip(next_state, STATE_MIN, STATE_MAX)

        reward = self.compute_reward(self.state, action, next_state)
        done = not self.is_safe(next_state, hard=True)

        self.state = next_state
        info = {
            "safe": self.is_safe(next_state, hard=False),
            "efficiency": self._efficiency(next_state),
        }
        return next_state.copy(), reward, done, info

    def is_safe(self, state: np.ndarray, hard: bool = True) -> bool:
        """
        检查状态安全性。

        Args:
            state: 状态向量
            hard:  True → 硬约束 (STATE_MIN/MAX), False → 报警约束
        """
        lo = STATE_MIN if hard else ALARM_MIN
        hi = STATE_MAX if hard else ALARM_MAX
        return bool(np.all(state >= lo) and np.all(state <= hi))

    def _efficiency(self, state: np.ndarray) -> float:
        """
        锅炉热效率估算 (0~1)。
        基于排烟温度、氧含量的简化 Siegert 公式。
        """
        T_fg = state[4]
        O2 = state[3]
        T_a = 20.0  # 环境温度

        # 排烟热损失
        q2 = (T_fg - T_a) * (0.66 + 0.009 * O2) / 100.0
        # 化学未完全燃烧损失 (O2 < 2.5% 时增大)
        q3 = max(0.0, (2.5 - O2) * 0.5) / 100.0
        # 机械未完全燃烧损失
        q4 = 0.005

        efficiency = 1.0 - q2 - q3 - q4
        return float(np.clip(efficiency, 0.7, 0.95))

    def compute_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> float:
        """
        奖励函数 = 热效率奖励 + 稳定性奖励 - 安全违约惩罚 - 动作幅度惩罚。
        """
        # 1. 热效率奖励
        eff = self._efficiency(next_state)
        reward_eff = 5.0 * eff

        # 2. 稳定性：与额定工况的偏差惩罚
        nominal = self._nominal_state()
        deviation = np.sum(
            ((next_state - nominal) / STATE_RANGE) ** 2
        )
        reward_stable = -0.5 * deviation

        # 3. 安全违约惩罚
        penalty_hard = 0.0
        penalty_alarm = 0.0
        for i in range(8):
            if next_state[i] < STATE_MIN[i] or next_state[i] > STATE_MAX[i]:
                penalty_hard += 10.0
            elif next_state[i] < ALARM_MIN[i] or next_state[i] > ALARM_MAX[i]:
                penalty_alarm += 2.0

        # 4. 动作平滑性惩罚
        action_penalty = -0.1 * np.sum(action ** 2)

        total = reward_eff + reward_stable - penalty_hard - penalty_alarm + action_penalty
        return float(total)


# ---------- 行为策略 ----------

class MixedBehaviorPolicy:
    """
    用于生成离线数据集的行为策略（混合策略）。

    由三类策略混合：
    - 50% 略优于随机（带向最优区间偏移）
    - 30% 纯随机探索
    - 20% 接近最优的确定性策略
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def act(self, state: np.ndarray) -> np.ndarray:
        policy_type = self.rng.random()
        if policy_type < 0.5:
            return self._heuristic(state)
        elif policy_type < 0.8:
            return self.rng.uniform(-1.0, 1.0, size=2)
        else:
            return self._near_optimal(state)

    def _heuristic(self, state: np.ndarray) -> np.ndarray:
        """启发式：朝最优目标小步移动，加随机扰动。"""
        target = (OPTIMAL_MIN + OPTIMAL_MAX) / 2.0
        diff = target - state
        coal_adj = np.clip(diff[5] / (2.0 * 10), -1.0, 1.0)
        air_adj = np.clip(diff[6] / (20.0 * 10), -1.0, 1.0)
        noise = self.rng.normal(0, 0.3, size=2)
        action = np.array([coal_adj, air_adj]) + noise
        return np.clip(action, -1.0, 1.0)

    def _near_optimal(self, state: np.ndarray) -> np.ndarray:
        """接近最优策略：微小调整。"""
        target = (OPTIMAL_MIN + OPTIMAL_MAX) / 2.0
        diff = target - state
        coal_adj = np.clip(diff[5] / (2.0 * 5), -1.0, 1.0)
        air_adj = np.clip(diff[6] / (20.0 * 5), -1.0, 1.0)
        noise = self.rng.normal(0, 0.1, size=2)
        action = np.array([coal_adj, air_adj]) + noise
        return np.clip(action, -1.0, 1.0)


# ---------- 数据集生成 ----------

def generate_offline_dataset(
    n_samples: int = 50000,
    episode_length: int = 200,
    noise_std: float = 0.02,
    random_seed: int = 42,
    save_dir: str = "dataset",
) -> dict:
    """
    生成离线强化学习数据集。

    Returns:
        dict with keys: states, actions, rewards, next_states, dones
        每个值为 shape (n_samples, dim) 的 np.ndarray
    """
    os.makedirs(save_dir, exist_ok=True)

    env = BoilerSimulator(noise_std=noise_std, random_seed=random_seed)
    rng = np.random.default_rng(random_seed)
    policy = MixedBehaviorPolicy(rng)

    states, actions, rewards, next_states, dones = [], [], [], [], []

    n_episodes = max(1, n_samples // episode_length + 1)
    collected = 0

    for ep in range(n_episodes):
        state = env.reset(random_init=True)
        for _ in range(episode_length):
            if collected >= n_samples:
                break
            action = policy.act(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state.copy())
            actions.append(action.copy())
            rewards.append(reward)
            next_states.append(next_state.copy())
            dones.append(float(done))

            collected += 1
            if done:
                state = env.reset(random_init=True)
            else:
                state = next_state
        if collected >= n_samples:
            break

    dataset = {
        "states": np.array(states[:n_samples], dtype=np.float32),
        "actions": np.array(actions[:n_samples], dtype=np.float32),
        "rewards": np.array(rewards[:n_samples], dtype=np.float32).reshape(-1, 1),
        "next_states": np.array(next_states[:n_samples], dtype=np.float32),
        "dones": np.array(dones[:n_samples], dtype=np.float32).reshape(-1, 1),
    }

    # 保存为 CSV（供调试和分析）
    df_state = pd.DataFrame(dataset["states"], columns=STATE_NAMES)
    df_action = pd.DataFrame(dataset["actions"], columns=ACTION_NAMES)
    df_reward = pd.DataFrame(dataset["rewards"], columns=["reward"])
    df_done = pd.DataFrame(dataset["dones"], columns=["done"])
    df = pd.concat([df_state, df_action, df_reward, df_done], axis=1)
    df.to_csv(os.path.join(save_dir, "offline_dataset.csv"), index=False)

    # 保存 numpy 格式
    np.save(os.path.join(save_dir, "states.npy"), dataset["states"])
    np.save(os.path.join(save_dir, "actions.npy"), dataset["actions"])
    np.save(os.path.join(save_dir, "rewards.npy"), dataset["rewards"])
    np.save(os.path.join(save_dir, "next_states.npy"), dataset["next_states"])
    np.save(os.path.join(save_dir, "dones.npy"), dataset["dones"])

    print(
        f"[DataGenerator] 数据集生成完毕: {collected} 条样本，"
        f"奖励均值={dataset['rewards'].mean():.4f}，"
        f"保存至 '{save_dir}/'"
    )
    return dataset


def load_offline_dataset(save_dir: str = "dataset") -> dict:
    """从磁盘加载已生成的离线数据集。"""
    dataset = {
        "states": np.load(os.path.join(save_dir, "states.npy")),
        "actions": np.load(os.path.join(save_dir, "actions.npy")),
        "rewards": np.load(os.path.join(save_dir, "rewards.npy")),
        "next_states": np.load(os.path.join(save_dir, "next_states.npy")),
        "dones": np.load(os.path.join(save_dir, "dones.npy")),
    }
    print(
        f"[DataGenerator] 数据集加载完毕: {len(dataset['states'])} 条样本，"
        f"来自 '{save_dir}/'"
    )
    return dataset


def split_dataset(
    dataset: dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple:
    """将数据集按比例分为训练/验证/测试集。"""
    n = len(dataset["states"])
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    def _split(arr, idx):
        return (
            arr[idx[:n_train]],
            arr[idx[n_train: n_train + n_val]],
            arr[idx[n_train + n_val:]],
        )

    train_ds, val_ds, test_ds = {}, {}, {}
    for key, val in dataset.items():
        tr, va, te = _split(val, idx)
        train_ds[key] = tr
        val_ds[key] = va
        test_ds[key] = te

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    dataset = generate_offline_dataset(n_samples=1000, save_dir="/tmp/test_dataset")
    train_ds, val_ds, test_ds = split_dataset(dataset)
    print(f"训练集: {len(train_ds['states'])}, 验证集: {len(val_ds['states'])}, 测试集: {len(test_ds['states'])}")

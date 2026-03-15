"""
utils/safety_constraints.py
安全约束检查与违约分析工具。

提供：
  - SafetyChecker：批量/单步约束检查
  - 违约统计报告
  - 运行轨迹安全评估
"""

import numpy as np
from data.data_generator import (
    STATE_NAMES,
    STATE_MIN,
    STATE_MAX,
    ALARM_MIN,
    ALARM_MAX,
    STATE_RANGE,
)


class SafetyChecker:
    """
    锅炉安全约束检查器。

    两级约束：
    - 硬约束 (hard)：物理上下限，违反 → 立即停止
    - 报警约束 (alarm)：运行优化区间，违反 → 报警但继续
    """

    def __init__(
        self,
        hard_min: np.ndarray = None,
        hard_max: np.ndarray = None,
        alarm_min: np.ndarray = None,
        alarm_max: np.ndarray = None,
    ):
        self.hard_min = hard_min if hard_min is not None else STATE_MIN.copy()
        self.hard_max = hard_max if hard_max is not None else STATE_MAX.copy()
        self.alarm_min = alarm_min if alarm_min is not None else ALARM_MIN.copy()
        self.alarm_max = alarm_max if alarm_max is not None else ALARM_MAX.copy()

    # ------------------------------------------------------------------
    # 单状态检查
    # ------------------------------------------------------------------

    def check_hard(self, state: np.ndarray) -> tuple:
        """
        检查硬约束。

        Returns:
            is_safe (bool), violations (dict)
        """
        violations = {}
        for i, name in enumerate(STATE_NAMES):
            if state[i] < self.hard_min[i]:
                violations[name] = (
                    "below_hard_min",
                    float(state[i]),
                    float(self.hard_min[i]),
                )
            elif state[i] > self.hard_max[i]:
                violations[name] = (
                    "above_hard_max",
                    float(state[i]),
                    float(self.hard_max[i]),
                )
        return len(violations) == 0, violations

    def check_alarm(self, state: np.ndarray) -> tuple:
        """
        检查报警约束。

        Returns:
            is_ok (bool), alarms (dict)
        """
        alarms = {}
        for i, name in enumerate(STATE_NAMES):
            if state[i] < self.alarm_min[i]:
                alarms[name] = (
                    "below_alarm_min",
                    float(state[i]),
                    float(self.alarm_min[i]),
                )
            elif state[i] > self.alarm_max[i]:
                alarms[name] = (
                    "above_alarm_max",
                    float(state[i]),
                    float(self.alarm_max[i]),
                )
        return len(alarms) == 0, alarms

    # ------------------------------------------------------------------
    # 批量检查
    # ------------------------------------------------------------------

    def batch_check_hard(self, states: np.ndarray) -> np.ndarray:
        """
        批量硬约束检查。

        Args:
            states: (N, state_dim)
        Returns:
            safe_mask: (N,) bool array
        """
        lo = self.hard_min[None, :]
        hi = self.hard_max[None, :]
        return np.all((states >= lo) & (states <= hi), axis=1)

    def batch_check_alarm(self, states: np.ndarray) -> np.ndarray:
        """批量报警约束检查，返回 (N,) bool。"""
        lo = self.alarm_min[None, :]
        hi = self.alarm_max[None, :]
        return np.all((states >= lo) & (states <= hi), axis=1)

    # ------------------------------------------------------------------
    # 违约率统计
    # ------------------------------------------------------------------

    def violation_rate(self, states: np.ndarray, level: str = "hard") -> dict:
        """
        统计各维度违约率。

        Args:
            states: (N, state_dim)
            level:  "hard" or "alarm"
        Returns:
            report: dict[name → violation_rate]
        """
        lo = self.hard_min if level == "hard" else self.alarm_min
        hi = self.hard_max if level == "hard" else self.alarm_max
        n = len(states)
        report = {}
        for i, name in enumerate(STATE_NAMES):
            below = np.sum(states[:, i] < lo[i])
            above = np.sum(states[:, i] > hi[i])
            report[name] = {
                "below_rate": below / n,
                "above_rate": above / n,
                "total_rate": (below + above) / n,
            }
        return report

    # ------------------------------------------------------------------
    # 轨迹安全评估
    # ------------------------------------------------------------------

    def evaluate_trajectory(self, states: np.ndarray) -> dict:
        """
        对完整运行轨迹进行安全评估。

        Args:
            states: (T, state_dim) 时间序列
        Returns:
            summary: 安全评估摘要
        """
        T = len(states)
        hard_safe = self.batch_check_hard(states)
        alarm_ok = self.batch_check_alarm(states)

        hard_viol_rate = 1.0 - hard_safe.mean()
        alarm_viol_rate = 1.0 - alarm_ok.mean()

        # 连续安全步数
        max_safe_run = 0
        cur_run = 0
        for s in hard_safe:
            if s:
                cur_run += 1
                max_safe_run = max(max_safe_run, cur_run)
            else:
                cur_run = 0

        # 归一化偏差（相对于STATE_RANGE）
        norm_states = (states - STATE_MIN[None, :]) / STATE_RANGE[None, :]
        center_norm = 0.5  # 中心
        avg_deviation = np.mean(np.abs(norm_states - center_norm))

        return {
            "total_steps": T,
            "hard_violation_rate": float(hard_viol_rate),
            "alarm_violation_rate": float(alarm_viol_rate),
            "max_safe_run_length": int(max_safe_run),
            "avg_norm_deviation": float(avg_deviation),
            "always_safe": bool(hard_safe.all()),
        }

    def print_report(self, states: np.ndarray, title: str = "安全评估报告") -> None:
        """打印人类可读的安全报告。"""
        summary = self.evaluate_trajectory(states)
        viol = self.violation_rate(states, level="hard")

        print(f"\n{'='*55}")
        print(f"  {title}")
        print(f"{'='*55}")
        print(f"  总步数：         {summary['total_steps']}")
        print(f"  硬约束违约率：   {summary['hard_violation_rate']:.2%}")
        print(f"  报警约束违约率： {summary['alarm_violation_rate']:.2%}")
        print(f"  最长安全运行：   {summary['max_safe_run_length']} 步")
        print(f"  平均归一化偏差： {summary['avg_norm_deviation']:.4f}")
        print(f"  全程安全：       {'✓' if summary['always_safe'] else '✗'}")
        print(f"\n  各维度硬约束违约率:")
        for name, info in viol.items():
            rate = info["total_rate"]
            if rate > 0:
                print(f"    {name:20s}: {rate:.2%}")
        print(f"{'='*55}\n")

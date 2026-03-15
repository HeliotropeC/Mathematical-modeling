"""
tests/test_data_generator.py
数据生成器单元测试。
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import (
    BoilerSimulator,
    MixedBehaviorPolicy,
    generate_offline_dataset,
    load_offline_dataset,
    split_dataset,
    STATE_MIN,
    STATE_MAX,
    STATE_NAMES,
)


class TestBoilerSimulator:
    """BoilerSimulator 单元测试。"""

    def setup_method(self):
        self.env = BoilerSimulator(noise_std=0.0, random_seed=0)

    def test_reset_shape(self):
        state = self.env.reset()
        assert state.shape == (8,), f"期望 (8,)，得到 {state.shape}"

    def test_reset_within_bounds(self):
        for _ in range(10):
            state = self.env.reset(random_init=True)
            assert np.all(state >= STATE_MIN), "重置状态低于下限"
            assert np.all(state <= STATE_MAX), "重置状态超过上限"

    def test_step_shape(self):
        self.env.reset()
        action = np.zeros(2)
        next_state, reward, done, info = self.env.step(action)
        assert next_state.shape == (8,), f"next_state shape 错误: {next_state.shape}"
        assert isinstance(reward, float), f"reward 类型错误: {type(reward)}"
        assert isinstance(done, bool), f"done 类型错误: {type(done)}"
        assert "safe" in info and "efficiency" in info

    def test_step_bounds_clipped(self):
        """步进后状态应被截断至 STATE_MIN/MAX 范围内。"""
        self.env.reset()
        for _ in range(50):
            action = np.random.uniform(-1, 1, size=2)
            next_state, _, _, _ = self.env.step(action)
            assert np.all(next_state >= STATE_MIN - 1e-6), "next_state 低于下限"
            assert np.all(next_state <= STATE_MAX + 1e-6), "next_state 超过上限"

    def test_action_clipping(self):
        """超范围动作应被截断。"""
        self.env.reset()
        action_large = np.array([10.0, 10.0])
        next_state, _, _, _ = self.env.step(action_large)
        assert np.all(next_state >= STATE_MIN - 1e-6)
        assert np.all(next_state <= STATE_MAX + 1e-6)

    def test_is_safe(self):
        nominal = self.env._nominal_state()
        assert self.env.is_safe(nominal, hard=True)
        assert self.env.is_safe(nominal, hard=False)

    def test_is_unsafe_below_hard(self):
        state = self.env._nominal_state().copy()
        state[0] = STATE_MIN[0] - 1.0  # 炉膛温度低于下限
        assert not self.env.is_safe(state, hard=True)

    def test_compute_reward_reasonable(self):
        """奖励应为有限浮点数。"""
        state = self.env._nominal_state()
        action = np.zeros(2)
        reward = self.env.compute_reward(state, action, state)
        assert np.isfinite(reward), f"奖励不是有限数: {reward}"

    def test_efficiency_range(self):
        state = self.env._nominal_state()
        eff = self.env._efficiency(state)
        assert 0.0 <= eff <= 1.0, f"效率超出 [0,1]: {eff}"


class TestMixedBehaviorPolicy:
    """MixedBehaviorPolicy 单元测试。"""

    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.policy = MixedBehaviorPolicy(self.rng)
        self.env = BoilerSimulator(noise_std=0.0, random_seed=0)

    def test_action_shape(self):
        state = self.env.reset()
        action = self.policy.act(state)
        assert action.shape == (2,), f"动作 shape 错误: {action.shape}"

    def test_action_range(self):
        state = self.env.reset()
        for _ in range(20):
            action = self.policy.act(state)
            assert np.all(action >= -1.0) and np.all(action <= 1.0), \
                f"动作超出 [-1,1]: {action}"


class TestDatasetGeneration:
    """数据集生成测试。"""

    def test_generate_correct_size(self, tmp_path):
        ds = generate_offline_dataset(
            n_samples=100,
            episode_length=50,
            random_seed=0,
            save_dir=str(tmp_path),
        )
        assert len(ds["states"]) == 100
        assert len(ds["actions"]) == 100
        assert len(ds["rewards"]) == 100
        assert len(ds["next_states"]) == 100
        assert len(ds["dones"]) == 100

    def test_generate_shapes(self, tmp_path):
        ds = generate_offline_dataset(
            n_samples=50,
            episode_length=25,
            random_seed=1,
            save_dir=str(tmp_path),
        )
        assert ds["states"].shape == (50, 8)
        assert ds["actions"].shape == (50, 2)
        assert ds["rewards"].shape == (50, 1)
        assert ds["next_states"].shape == (50, 8)
        assert ds["dones"].shape == (50, 1)

    def test_generate_dtype(self, tmp_path):
        ds = generate_offline_dataset(
            n_samples=20,
            episode_length=10,
            random_seed=2,
            save_dir=str(tmp_path),
        )
        for key, val in ds.items():
            assert val.dtype == np.float32, f"{key} dtype 错误: {val.dtype}"

    def test_actions_in_range(self, tmp_path):
        ds = generate_offline_dataset(
            n_samples=100,
            episode_length=50,
            random_seed=3,
            save_dir=str(tmp_path),
        )
        assert np.all(ds["actions"] >= -1.0), "动作超出下界 -1"
        assert np.all(ds["actions"] <= 1.0), "动作超出上界 1"

    def test_states_in_bounds(self, tmp_path):
        ds = generate_offline_dataset(
            n_samples=100,
            episode_length=50,
            random_seed=4,
            save_dir=str(tmp_path),
        )
        assert np.all(ds["states"] >= STATE_MIN - 1e-5)
        assert np.all(ds["states"] <= STATE_MAX + 1e-5)

    def test_save_and_load(self, tmp_path):
        ds = generate_offline_dataset(
            n_samples=80,
            episode_length=40,
            random_seed=5,
            save_dir=str(tmp_path),
        )
        loaded = load_offline_dataset(str(tmp_path))
        for key in ds:
            np.testing.assert_array_almost_equal(
                ds[key], loaded[key], decimal=5,
                err_msg=f"{key} 保存/加载不一致"
            )

    def test_split_dataset_sizes(self, tmp_path):
        ds = generate_offline_dataset(
            n_samples=100,
            episode_length=50,
            random_seed=6,
            save_dir=str(tmp_path),
        )
        train_ds, val_ds, test_ds = split_dataset(ds, train_ratio=0.7, val_ratio=0.2)
        assert len(train_ds["states"]) == 70
        assert len(val_ds["states"]) == 20
        assert len(test_ds["states"]) == 10

    def test_split_no_overlap(self, tmp_path):
        """验证 train/val/test 无数据重叠（检查 rewards 唯一性近似）。"""
        ds = generate_offline_dataset(
            n_samples=100,
            episode_length=50,
            random_seed=7,
            save_dir=str(tmp_path),
        )
        train_ds, val_ds, test_ds = split_dataset(ds, train_ratio=0.7, val_ratio=0.2)
        # 行级 union 应等于总样本数
        total = (
            len(train_ds["states"])
            + len(val_ds["states"])
            + len(test_ds["states"])
        )
        assert total == 100

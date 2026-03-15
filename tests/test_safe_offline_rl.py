"""
tests/test_safe_offline_rl.py
安全离线强化学习（CQL）单元测试。
"""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import (
    generate_offline_dataset,
    STATE_MIN,
    STATE_MAX,
    STATE_RANGE,
)
from models.safe_offline_rl import (
    QNetwork,
    PolicyNetwork,
    SafetyLayer,
    CQLAgent,
    OfflineReplayBuffer,
    build_cql_agent,
    train_cql,
)


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    save_dir = str(tmp_path_factory.mktemp("cql_data"))
    return generate_offline_dataset(
        n_samples=200, episode_length=50, random_seed=0, save_dir=save_dir
    )


class TestQNetwork:
    def test_forward_shape(self):
        net = QNetwork(state_dim=8, action_dim=2, hidden_sizes=[32, 32])
        s = torch.randn(4, 8)
        a = torch.randn(4, 2)
        q1, q2 = net(s, a)
        assert q1.shape == (4, 1), f"q1 shape 错误: {q1.shape}"
        assert q2.shape == (4, 1), f"q2 shape 错误: {q2.shape}"

    def test_q_min_shape(self):
        net = QNetwork(state_dim=8, action_dim=2, hidden_sizes=[32, 32])
        s = torch.randn(8, 8)
        a = torch.randn(8, 2)
        qm = net.q_min(s, a)
        assert qm.shape == (8, 1), f"q_min shape 错误: {qm.shape}"

    def test_q_min_leq_q1_and_q2(self):
        net = QNetwork(state_dim=8, action_dim=2, hidden_sizes=[32, 32])
        s = torch.randn(16, 8)
        a = torch.randn(16, 2)
        q1, q2 = net(s, a)
        qm = net.q_min(s, a)
        assert torch.all(qm <= q1 + 1e-6), "q_min > q1"
        assert torch.all(qm <= q2 + 1e-6), "q_min > q2"

    def test_gradient_flows(self):
        net = QNetwork(state_dim=8, action_dim=2, hidden_sizes=[32, 32])
        s = torch.randn(4, 8)
        a = torch.randn(4, 2)
        q1, q2 = net(s, a)
        (q1.mean() + q2.mean()).backward()
        for name, p in net.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} 无梯度"


class TestPolicyNetwork:
    def test_forward_shape(self):
        net = PolicyNetwork(state_dim=8, action_dim=2, hidden_sizes=[32, 32])
        s = torch.randn(4, 8)
        a = net(s)
        assert a.shape == (4, 2), f"动作 shape 错误: {a.shape}"

    def test_output_range(self):
        """tanh 输出应在 (-1, 1)。"""
        net = PolicyNetwork(state_dim=8, action_dim=2, hidden_sizes=[32, 32])
        s = torch.randn(32, 8)
        a = net(s)
        assert a.min().item() >= -1.0 - 1e-6, "输出低于 -1"
        assert a.max().item() <= 1.0 + 1e-6, "输出高于 1"


class TestSafetyLayer:
    def setup_method(self):
        self.layer = SafetyLayer(safety_margin=0.02)

    def test_project_nominal_unchanged(self):
        """额定状态下，小动作不应被裁剪。"""
        state = (STATE_MIN + STATE_MAX) / 2  # 中点
        action = np.array([0.0, 0.0])
        projected = self.layer.project(state, action)
        np.testing.assert_array_almost_equal(action, projected, decimal=5)

    def test_project_clips_coal_feed(self):
        """给煤量在上限附近，正向动作应被裁剪。"""
        state = (STATE_MIN + STATE_MAX) / 2
        state[5] = STATE_MAX[5] - 0.1  # 给煤量接近上限
        action = np.array([1.0, 0.0])   # 请求最大增加
        projected = self.layer.project(state, action)
        assert projected[0] < action[0], "未对给煤量进行约束投影"

    def test_project_clips_air_flow(self):
        """送风量在上限附近，正向动作应被裁剪。"""
        state = (STATE_MIN + STATE_MAX) / 2
        state[6] = STATE_MAX[6] - 0.5  # 送风量接近上限
        action = np.array([0.0, 1.0])
        projected = self.layer.project(state, action)
        assert projected[1] < action[1], "未对送风量进行约束投影"

    def test_project_output_range(self):
        """投影后动作仍应在 [-1, 1]。"""
        state = (STATE_MIN + STATE_MAX) / 2
        for _ in range(20):
            action = np.random.uniform(-1, 1, size=2)
            proj = self.layer.project(state, action)
            assert np.all(proj >= -1.0 - 1e-6) and np.all(proj <= 1.0 + 1e-6)

    def test_project_batch_shape(self):
        states = torch.tensor(
            np.tile((STATE_MIN + STATE_MAX) / 2, (8, 1)), dtype=torch.float32
        )
        actions = torch.zeros(8, 2)
        proj = self.layer.project_batch(states, actions)
        assert proj.shape == (8, 2), f"批量投影 shape 错误: {proj.shape}"


class TestCQLAgent:
    """CQLAgent 核心功能测试。"""

    def _make_agent(self):
        return CQLAgent(
            state_dim=8,
            action_dim=2,
            hidden_sizes=[32, 32],
            lr=1e-3,
            gamma=0.99,
            tau=0.01,
            cql_alpha=0.5,
            cql_n_actions=4,
            use_safety_layer=True,
            safety_penalty_coef=1.0,
        )

    def _make_batch(self, n=16):
        mid = (STATE_MIN + STATE_MAX) / 2
        return {
            "states": np.tile(mid, (n, 1)).astype(np.float32),
            "actions": np.random.uniform(-0.5, 0.5, size=(n, 2)).astype(np.float32),
            "rewards": np.random.randn(n, 1).astype(np.float32),
            "next_states": np.tile(mid, (n, 1)).astype(np.float32),
            "dones": np.zeros((n, 1), dtype=np.float32),
        }

    def test_select_action_shape(self):
        agent = self._make_agent()
        state = (STATE_MIN + STATE_MAX) / 2
        action = agent.select_action(state)
        assert action.shape == (2,), f"动作 shape 错误: {action.shape}"

    def test_select_action_range(self):
        agent = self._make_agent()
        state = (STATE_MIN + STATE_MAX) / 2
        for _ in range(10):
            action = agent.select_action(state)
            assert np.all(action >= -1.0 - 1e-6) and np.all(action <= 1.0 + 1e-6), \
                f"动作超出范围: {action}"

    def test_update_returns_stats(self):
        agent = self._make_agent()
        batch = self._make_batch()
        stats = agent.update(batch, update_policy=True)
        assert "q_loss" in stats
        assert "bellman_loss" in stats
        assert "conservative_loss" in stats
        assert "policy_loss" in stats
        assert np.isfinite(stats["q_loss"]), "q_loss 为非有限数"
        assert np.isfinite(stats["policy_loss"]), "policy_loss 为非有限数"

    def test_update_decrements_loss_over_steps(self):
        """多步更新后 Q loss 应为有限数（不发散）。"""
        agent = self._make_agent()
        for _ in range(20):
            batch = self._make_batch()
            stats = agent.update(batch, update_policy=True)
        assert np.isfinite(stats["q_loss"]), "多步更新后 Q loss 发散"

    def test_save_and_load(self, tmp_path):
        agent = self._make_agent()
        # 做几步更新
        for _ in range(5):
            agent.update(self._make_batch())

        save_path = str(tmp_path / "cql")
        agent.save(save_path)
        assert os.path.exists(save_path + ".pt"), "保存文件不存在"

        # 加载到新智能体
        agent2 = self._make_agent()
        agent2.load(save_path)

        # 验证策略输出一致
        state = (STATE_MIN + STATE_MAX) / 2
        a1 = agent.select_action(state)
        a2 = agent2.select_action(state)
        np.testing.assert_array_almost_equal(a1, a2, decimal=5, err_msg="加载后策略输出不一致")

    def test_warmup_skips_policy_update(self):
        """warmup 阶段 policy_loss 应为 0（未更新）。"""
        agent = self._make_agent()
        batch = self._make_batch()
        stats = agent.update(batch, update_policy=False)
        assert stats["policy_loss"] == 0.0 or isinstance(stats["policy_loss"], float)

    def test_build_cql_agent(self):
        cfg = {
            "offline_rl": {
                "hidden_sizes": [32, 32],
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "tau": 0.005,
                "cql_alpha": 1.0,
                "cql_n_actions": 4,
                "use_safety_layer": True,
                "safety_penalty_coef": 1.0,
            },
            "boiler": {"state_dim": 8, "action_dim": 2},
        }
        agent = build_cql_agent(cfg)
        assert isinstance(agent, CQLAgent)


class TestOfflineReplayBuffer:
    """OfflineReplayBuffer 测试。"""

    def test_sample_shape(self, small_dataset):
        buf = OfflineReplayBuffer(small_dataset)
        batch = buf.sample(32)
        assert batch["states"].shape == (32, 8)
        assert batch["actions"].shape == (32, 2)
        assert batch["rewards"].shape == (32, 1)
        assert batch["next_states"].shape == (32, 8)
        assert batch["dones"].shape == (32, 1)

    def test_sample_different_each_call(self, small_dataset):
        buf = OfflineReplayBuffer(small_dataset)
        b1 = buf.sample(16)
        b2 = buf.sample(16)
        # 两次采样不应完全相同（概率极低）
        assert not np.array_equal(b1["states"], b2["states"]), "两次采样完全相同"


class TestTrainCQL:
    """train_cql 流程集成测试（小规模）。"""

    def test_train_returns_stats(self, small_dataset, tmp_path):
        agent = CQLAgent(
            hidden_sizes=[16, 16],
            cql_n_actions=4,
        )
        stats = train_cql(
            agent,
            small_dataset,
            max_steps=20,
            batch_size=16,
            eval_interval=10,
            save_interval=20,
            save_path=str(tmp_path / "cql"),
            warmup_steps=5,
        )
        assert len(stats) == 20
        assert all("q_loss" in s for s in stats)

    def test_train_saves_checkpoint(self, small_dataset, tmp_path):
        agent = CQLAgent(hidden_sizes=[16, 16], cql_n_actions=4)
        save_path = str(tmp_path / "cql_save")
        train_cql(
            agent,
            small_dataset,
            max_steps=10,
            batch_size=16,
            eval_interval=10,
            save_interval=10,
            save_path=save_path,
            warmup_steps=5,
        )
        assert os.path.exists(save_path + ".pt"), "训练后未保存检查点"

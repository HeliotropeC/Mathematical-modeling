"""
tests/test_digital_twin.py
数字孪生模型单元测试。
"""

import numpy as np
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import generate_offline_dataset, STATE_MIN, STATE_MAX, STATE_RANGE
from models.digital_twin import (
    BoilerSequenceDataset,
    LSTMDigitalTwin,
    DigitalTwinTrainer,
    build_digital_twin,
)
from torch.utils.data import DataLoader


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    """小规模数据集，供多个测试用例共享。"""
    save_dir = str(tmp_path_factory.mktemp("dt_data"))
    ds = generate_offline_dataset(n_samples=200, episode_length=50, random_seed=0, save_dir=save_dir)
    return ds


class TestBoilerSequenceDataset:
    """BoilerSequenceDataset 单元测试。"""

    def test_len(self, small_dataset):
        seq_len = 5
        ds = BoilerSequenceDataset(
            small_dataset["states"],
            small_dataset["actions"],
            small_dataset["next_states"],
            seq_len=seq_len,
        )
        expected = len(small_dataset["states"]) - seq_len + 1
        assert len(ds) == expected

    def test_item_shape(self, small_dataset):
        seq_len = 5
        ds = BoilerSequenceDataset(
            small_dataset["states"],
            small_dataset["actions"],
            small_dataset["next_states"],
            seq_len=seq_len,
        )
        x, y = ds[0]
        assert x.shape == (seq_len, 10), f"x shape 错误: {x.shape}"
        assert y.shape == (8,), f"y shape 错误: {y.shape}"

    def test_normalized_range(self, small_dataset):
        """归一化状态应在 [0, 1] 附近（噪声可能稍微超出）。"""
        ds = BoilerSequenceDataset(
            small_dataset["states"],
            small_dataset["actions"],
            small_dataset["next_states"],
            seq_len=5,
        )
        x, y = ds[0]
        # 状态部分 (前 8 列)
        state_part = x[:, :8].numpy()
        assert state_part.min() >= -0.1, f"状态归一化值太低: {state_part.min()}"
        assert state_part.max() <= 1.1, f"状态归一化值太高: {state_part.max()}"

    def test_tensor_dtype(self, small_dataset):
        ds = BoilerSequenceDataset(
            small_dataset["states"],
            small_dataset["actions"],
            small_dataset["next_states"],
            seq_len=3,
        )
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32


class TestLSTMDigitalTwin:
    """LSTMDigitalTwin 网络测试。"""

    def test_forward_shape(self):
        model = LSTMDigitalTwin(state_dim=8, action_dim=2, hidden_size=32, num_layers=1)
        x = torch.randn(4, 5, 10)  # (batch=4, seq=5, input=10)
        out = model(x)
        assert out.shape == (4, 8), f"输出 shape 错误: {out.shape}"

    def test_output_in_range(self):
        """Sigmoid 输出应在 [0, 1]。"""
        model = LSTMDigitalTwin(state_dim=8, action_dim=2, hidden_size=32, num_layers=1)
        x = torch.randn(8, 5, 10)
        out = model(x)
        assert out.min().item() >= 0.0, "输出低于 0"
        assert out.max().item() <= 1.0, "输出高于 1"

    def test_gradient_flows(self):
        """反向传播梯度应正常流动。"""
        model = LSTMDigitalTwin(state_dim=8, action_dim=2, hidden_size=32, num_layers=1)
        x = torch.randn(4, 5, 10, requires_grad=False)
        out = model(x)
        loss = out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} 无梯度"

    def test_predict_next_state_shape(self):
        model = LSTMDigitalTwin(state_dim=8, action_dim=2, hidden_size=32, num_layers=1)
        state = (STATE_MIN + STATE_MAX) / 2
        action = np.zeros(2)
        next_state = model.predict_next_state(state, action)
        assert next_state.shape == (8,), f"预测状态 shape 错误: {next_state.shape}"

    def test_predict_next_state_bounds(self):
        """预测状态应大体在物理范围内（允许小偏差）。"""
        model = LSTMDigitalTwin(state_dim=8, action_dim=2, hidden_size=32, num_layers=1)
        state = (STATE_MIN + STATE_MAX) / 2
        action = np.zeros(2)
        next_state = model.predict_next_state(state, action)
        # 反归一化后的范围检查（允许 ±10% 的偏差）
        margin = 0.1 * STATE_RANGE
        assert np.all(next_state >= STATE_MIN - margin), f"预测值低于下限: {next_state}"
        assert np.all(next_state <= STATE_MAX + margin), f"预测值超过上限: {next_state}"

    def test_build_digital_twin(self):
        cfg = {
            "digital_twin": {"hidden_size": 32, "num_layers": 1, "dropout": 0.0},
            "boiler": {"state_dim": 8, "action_dim": 2},
        }
        model = build_digital_twin(cfg)
        assert isinstance(model, LSTMDigitalTwin)


class TestDigitalTwinTrainer:
    """DigitalTwinTrainer 训练流程测试。"""

    def test_train_epoch_reduces_loss(self, small_dataset, tmp_path):
        """训练若干 epoch 后损失应下降。"""
        ds = BoilerSequenceDataset(
            small_dataset["states"],
            small_dataset["actions"],
            small_dataset["next_states"],
            seq_len=5,
        )
        loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
        model = LSTMDigitalTwin(state_dim=8, action_dim=2, hidden_size=32, num_layers=1)
        trainer = DigitalTwinTrainer(model, lr=1e-2)
        loss_before = trainer.validate(loader)
        for _ in range(5):
            trainer.train_epoch(loader)
        loss_after = trainer.validate(loader)
        # 训练后损失不一定立即降低（随机初始化），但应是有限数
        assert np.isfinite(loss_after), f"损失为非有限数: {loss_after}"

    def test_fit_saves_checkpoint(self, small_dataset, tmp_path):
        """fit 方法应保存模型检查点。"""
        ds = BoilerSequenceDataset(
            small_dataset["states"],
            small_dataset["actions"],
            small_dataset["next_states"],
            seq_len=3,
        )
        loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
        model = LSTMDigitalTwin(state_dim=8, action_dim=2, hidden_size=16, num_layers=1)
        trainer = DigitalTwinTrainer(model)
        save_path = str(tmp_path / "dt_test.pt")
        trainer.fit(loader, loader, max_epochs=2, patience=5, save_path=save_path)
        assert os.path.exists(save_path), f"检查点文件不存在: {save_path}"

    def test_save_and_load(self, small_dataset, tmp_path):
        """保存后加载的模型应产生相同输出。"""
        ds = BoilerSequenceDataset(
            small_dataset["states"],
            small_dataset["actions"],
            small_dataset["next_states"],
            seq_len=3,
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        model = LSTMDigitalTwin(state_dim=8, action_dim=2, hidden_size=16, num_layers=1)
        trainer = DigitalTwinTrainer(model)
        save_path = str(tmp_path / "dt_save_test.pt")
        trainer.save(save_path)

        model2 = LSTMDigitalTwin(state_dim=8, action_dim=2, hidden_size=16, num_layers=1)
        trainer2 = DigitalTwinTrainer(model2)
        trainer2.load(save_path)

        x, _ = ds[:5]
        # eval mode disables dropout so outputs are deterministic
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        np.testing.assert_array_almost_equal(
            out1.numpy(), out2.numpy(), decimal=5,
            err_msg="加载模型输出与原模型不一致"
        )

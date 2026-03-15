"""
models/digital_twin.py
基于 LSTM 的锅炉数字孪生模型。

功能：
  - 接受当前状态 + 动作 → 预测下一状态
  - 支持多步滚动预测（序列输入）
  - 提供归一化 / 反归一化接口
  - 支持模型保存 / 加载
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.data_generator import STATE_MIN, STATE_RANGE

# ---------- 数据集 ----------

class BoilerSequenceDataset(Dataset):
    """
    将 (state, action, next_state) 数据集组织为序列输入。

    对每个样本构建长度为 seq_len 的历史窗口，输入为
    [s_{t-seq_len+1}, ..., s_t, a_{t-seq_len+1}, ..., a_t]，
    目标为 s_{t+1}。
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        seq_len: int = 10,
    ):
        super().__init__()
        self.seq_len = seq_len

        # 归一化到 [0, 1]
        states_norm = (states - STATE_MIN) / STATE_RANGE
        next_norm = (next_states - STATE_MIN) / STATE_RANGE

        n = len(states)
        # 构建有效样本（需有足够历史）
        self.x_list = []
        self.y_list = []

        for i in range(seq_len - 1, n):
            # 历史状态和动作
            hist_states = states_norm[i - seq_len + 1: i + 1]   # (seq_len, 8)
            hist_actions = actions[i - seq_len + 1: i + 1]       # (seq_len, 2)
            x = np.concatenate([hist_states, hist_actions], axis=-1)  # (seq_len, 10)
            y = next_norm[i]                                          # (8,)
            self.x_list.append(x)
            self.y_list.append(y)

        self.x = torch.tensor(np.array(self.x_list), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y_list), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# ---------- LSTM 数字孪生网络 ----------

class LSTMDigitalTwin(nn.Module):
    """
    LSTM 数字孪生网络。

    输入: (batch, seq_len, state_dim + action_dim)
    输出: (batch, state_dim) — 预测的下一时刻归一化状态
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        input_size = state_dim + action_dim

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, state_dim),
            nn.Sigmoid(),  # 输出限制在 [0, 1]（归一化状态）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, state_dim + action_dim)
        Returns:
            next_state_norm: (batch, state_dim)
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last_out = lstm_out[:, -1, :]       # 取最后时间步
        return self.head(last_out)          # (batch, state_dim)

    def predict_next_state(
        self,
        state: np.ndarray,
        action: np.ndarray,
        history: np.ndarray = None,
    ) -> np.ndarray:
        """
        单步预测（numpy 接口）。

        Args:
            state:   当前状态 (state_dim,) 原始尺度
            action:  归一化动作 (action_dim,)
            history: 可选历史 (seq_len-1, state_dim + action_dim) 归一化
        Returns:
            next_state: (state_dim,) 原始尺度
        """
        self.eval()
        state_norm = (state - STATE_MIN) / STATE_RANGE
        cur = np.concatenate([state_norm, action])  # (10,)

        if history is None or len(history) == 0:
            seq_len = self.lstm.num_layers + 1
            hist_pad = np.zeros((seq_len - 1, self.state_dim + self.action_dim))
            x = np.concatenate([hist_pad, cur[None, :]], axis=0)[None, :, :]
        else:
            x = np.concatenate([history, cur[None, :]], axis=0)[None, :, :]

        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            pred_norm = self(x_t).numpy()[0]  # (state_dim,)

        return pred_norm * STATE_RANGE + STATE_MIN  # 反归一化


# ---------- 训练器 ----------

class DigitalTwinTrainer:
    """封装数字孪生模型的训练、验证和保存逻辑。"""

    def __init__(
        self,
        model: LSTMDigitalTwin,
        lr: float = 1e-3,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=5
        )
        self.grad_clip = grad_clip
        self.criterion = nn.MSELoss()
        self.train_losses: list = []
        self.val_losses: list = []

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * len(x)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            total_loss += loss.item() * len(x)
        return total_loss / len(loader.dataset)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        patience: int = 15,
        save_path: str = "checkpoints/digital_twin.pt",
    ) -> dict:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        best_val = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, max_epochs + 1):
            tr_loss = self.train_epoch(train_loader)
            va_loss = self.validate(val_loader)
            self.scheduler.step(va_loss)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"[DigitalTwin] Epoch {epoch:4d}/{max_epochs} | "
                    f"Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f}"
                )

            if va_loss < best_val:
                best_val = va_loss
                patience_counter = 0
                self.save(save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[DigitalTwin] 早停于 Epoch {epoch}，最佳验证损失: {best_val:.6f}")
                    break

        self.train_losses = history["train_loss"]
        self.val_losses = history["val_loss"]
        print(f"[DigitalTwin] 训练完成，模型保存至 '{save_path}'")
        return history

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()


def build_digital_twin(cfg: dict) -> LSTMDigitalTwin:
    """根据配置字典构建数字孪生模型。"""
    dt_cfg = cfg.get("digital_twin", {})
    boiler_cfg = cfg.get("boiler", {})
    return LSTMDigitalTwin(
        state_dim=boiler_cfg.get("state_dim", 8),
        action_dim=boiler_cfg.get("action_dim", 2),
        hidden_size=dt_cfg.get("hidden_size", 128),
        num_layers=dt_cfg.get("num_layers", 2),
        dropout=dt_cfg.get("dropout", 0.1),
    )


if __name__ == "__main__":
    from data.data_generator import generate_offline_dataset

    ds = generate_offline_dataset(n_samples=500, save_dir="/tmp/dt_test")
    seq_ds = BoilerSequenceDataset(ds["states"], ds["actions"], ds["next_states"], seq_len=5)
    loader = DataLoader(seq_ds, batch_size=32, shuffle=True)
    model = LSTMDigitalTwin()
    trainer = DigitalTwinTrainer(model)
    trainer.fit(loader, loader, max_epochs=3, save_path="/tmp/dt_test/dt.pt")
    print("数字孪生模型测试通过。")

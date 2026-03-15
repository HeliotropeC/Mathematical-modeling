# Mathematical-modeling

记录数学建模学习过程

---

9.7  高社杯全国大学生数学建模比赛结束

## 收获：

1. 学会了 copilot 和 cursor 的代理模型，对于代码修改与运行提供极大便利
2. 进一步熟练建模过程书写与处理
3. 复习了 latex 文件排版，包含文件引用等
4. 更高级的可视化模型，如平行图等

---

## 需要改进：

1. ai 指令需要完整与清晰的表示，分段化的表述对结果有较大影响
2. 在处理问题时明确题目要求的输入与输出，重点关注而不是其他可以优化的内容
3. 先用代码写结果，再给建模过程，否则在书写时难以出结果图
4. 保持充足的睡眠以确保高效率，而不是熬时间

---

# 燃煤锅炉运行参数安全离线优化系统

本项目实现了一套完整的 **燃煤锅炉运行参数安全离线优化系统**，融合了时间序列建模、数字孪生和安全离线强化学习三大技术。

## 目录结构

```
Mathematical-modeling/
├── config/
│   └── config.yaml            # 全局配置（数据、模型、训练超参）
├── data/
│   ├── __init__.py
│   └── data_generator.py      # 锅炉仿真环境 & 离线数据集生成器
├── models/
│   ├── __init__.py
│   ├── digital_twin.py        # LSTM 数字孪生模型
│   └── safe_offline_rl.py     # 安全离线强化学习 (CQL)
├── utils/
│   ├── __init__.py
│   ├── safety_constraints.py  # 安全约束检查器
│   └── visualization.py       # 训练与评估可视化
├── train/
│   ├── __init__.py
│   ├── train_digital_twin.py  # 数字孪生训练脚本
│   └── train_offline_rl.py    # 离线 RL 训练脚本
├── evaluate/
│   ├── __init__.py
│   └── evaluate.py            # 多策略对比评估脚本
├── tests/
│   ├── __init__.py
│   ├── test_data_generator.py
│   ├── test_digital_twin.py
│   └── test_safe_offline_rl.py
├── main.py                    # 统一入口
├── requirements.txt
└── README.md
```

## 系统架构

```
离线历史数据
      │
      ▼
┌─────────────────┐     ┌──────────────────────┐
│  锅炉仿真环境    │────▶│  离线数据集           │
│  BoilerSimulator│     │  (states, actions,    │
└─────────────────┘     │   rewards, next_states│
                        └──────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
          ┌──────────────────┐       ┌────────────────────┐
          │  数字孪生         │       │  安全离线 RL (CQL) │
          │  LSTM 模型        │       │  - Q 网络 (Double) │
          │  状态转移预测     │       │  - 策略网络 (Actor) │
          └──────────────────┘       │  - 保守正则化       │
                                     │  - 安全层投影       │
                                     └────────────────────┘
                                               │
                                               ▼
                                     ┌──────────────────┐
                                     │  安全约束检查器   │
                                     │  硬约束 / 报警    │
                                     └──────────────────┘
```

## 状态与动作空间

**状态空间（8 维）**

| 维度 | 名称       | 单位   | 下限  | 上限  |
|------|-----------|--------|-------|-------|
| 0    | 炉膛温度   | °C     | 900   | 1200  |
| 1    | 蒸汽压力   | MPa    | 8.0   | 14.0  |
| 2    | 蒸汽温度   | °C     | 500   | 560   |
| 3    | 氧含量     | %      | 2.0   | 6.0   |
| 4    | 排烟温度   | °C     | 120   | 200   |
| 5    | 给煤量     | t/h    | 20    | 40    |
| 6    | 送风量     | m³/h   | 200   | 400   |
| 7    | NOx 排放  | mg/m³  | 100   | 400   |

**动作空间（2 维，归一化到 `[-1, 1]`）**

| 维度 | 名称       | 实际范围   |
|------|-----------|-----------|
| 0    | 给煤量增量 | ±2 t/h    |
| 1    | 送风量增量 | ±20 m³/h  |

## 算法

### 数字孪生：LSTM 时序模型

- 输入：历史 `seq_len` 步的 `(状态, 动作)` 序列
- 输出：下一时刻状态预测（归一化，Sigmoid 激活）
- 训练：MSE 损失 + 早停 + 学习率自适应衰减

### 安全离线 RL：Conservative Q-Learning (CQL)

核心思想：在标准 Bellman 误差之外加入保守正则项，惩罚数据集外动作的 Q 值：

```
L_CQL(Q) = α·(log Σ_a exp(Q(s,a)) - Q(s,a_dataset))
         + E[(Q(s,a) - (r + γ·Q'(s',π(s'))))²] / 2
```

**安全层（Safety Layer）**：在策略输出后投影到不违反约束的可行动作域，防止给煤量/送风量超出安全范围。

## 安装

```bash
# 建议使用 Python 3.9+
pip install -r requirements.txt
```

## 使用说明

### 方式一：统一入口（推荐）

```bash
# 运行完整流程（数据生成 → 训练数字孪生 → 训练 RL → 评估）
python main.py all

# 仅生成数据集
python main.py generate

# 仅训练数字孪生
python main.py train-dt

# 仅训练离线 RL
python main.py train-rl

# 仅评估
python main.py evaluate

# 指定配置文件
python main.py all --config config/config.yaml
```

### 方式二：分步执行

```bash
# 1. 生成离线数据集
python -m data.data_generator

# 2. 训练数字孪生模型
python -m train.train_digital_twin --config config/config.yaml

# 3. 训练安全离线 RL 智能体
python -m train.train_offline_rl --config config/config.yaml

# 4. 评估与对比
python -m evaluate.evaluate --config config/config.yaml
```

## 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单个测试模块
pytest tests/test_data_generator.py -v
pytest tests/test_digital_twin.py -v
pytest tests/test_safe_offline_rl.py -v
```

## 输出文件

训练和评估完成后，结果保存在以下位置：

```
dataset/                     # 离线数据集（.npy + .csv）
checkpoints/
├── digital_twin.pt          # 数字孪生模型权重
└── cql_agent.pt             # CQL 智能体权重
results/
├── dt_training_curve.png    # 数字孪生训练损失曲线
├── dt_prediction.png        # 数字孪生预测 vs 真实对比
├── cql_training_stats.png   # CQL 训练统计（Q/Policy/Conservative Loss）
├── reward_comparison.png    # 多策略奖励对比
├── cql_trajectory.png       # CQL 策略状态轨迹
├── cql_safety_heatmap.png   # CQL 策略安全约束热图
├── heuristic_trajectory.png # 启发式策略状态轨迹
└── evaluation_summary.csv   # 评估数值汇总
```

## 配置说明

主要超参数均在 `config/config.yaml` 中集中管理：

```yaml
data:
  n_samples: 50000       # 离线数据集大小
  episode_length: 200    # 每 episode 步数

digital_twin:
  hidden_size: 128       # LSTM 隐藏层维度
  num_layers: 2          # LSTM 层数
  seq_len: 10            # 历史窗口长度
  max_epochs: 100        # 最大训练轮次

offline_rl:
  cql_alpha: 1.0         # CQL 保守正则化系数
  gamma: 0.99            # 折扣因子
  max_steps: 100000      # 离线训练总步数
  use_safety_layer: true # 是否启用安全层
```

## 依赖

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy ≥ 1.23
- Pandas ≥ 1.5
- Scikit-learn ≥ 1.2
- Matplotlib ≥ 3.6
- PyYAML ≥ 6.0
- tqdm ≥ 4.64
- pytest ≥ 7.2


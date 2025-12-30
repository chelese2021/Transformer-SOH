# 电池SOC和SOH智能预测系统

基于Transformer深度学习模型的电池状态实时分析系统，用于精确预测电池的SOC（充电状态）和SOH（健康状态）。

## 📋 项目概述

本项目使用Transformer神经网络对电池的电压、电流、温度等关键数据进行时间序列分析，实现：

- **SOC预测**：实时评估电池剩余电量（0-100%）
- **SOH预测**：准确评估电池健康状态（0-1，越接近1越健康）
- **实时分析**：基于历史10分钟数据（60个时间步×10秒）进行预测

## 🏗️ 系统架构

### 模型架构
```
输入特征 (6维)
    ↓
输入嵌入层 (Linear)
    ↓
位置编码 (Positional Encoding)
    ↓
Transformer编码器 (4层)
  - 多头注意力机制 (8个头)
  - 前馈神经网络 (512维)
    ↓
全局平均池化
    ↓
双输出头
  ├─ SOC预测头
  └─ SOH预测头
    ↓
输出 (SOC, SOH)
```

### 模型参数
- **输入维度**: 6（充电电流、最大/最小单体电压、最大/最小单体温度、里程）
- **模型维度**: 128
- **注意力头数**: 8
- **编码器层数**: 4
- **前馈网络维度**: 512
- **总参数量**: ~400,000

## 📊 数据集

- **数据来源**: 100个CSV文件，约1700万条记录
- **特征维度**: 9维（6个输入特征 + 3个辅助特征）
- **时间采样**: 每10秒一条记录
- **数据划分**: 训练集70%、验证集15%、测试集15%

### 特征说明
| 特征名称 | 说明 | 单位 |
|---------|------|------|
| Charging_Current | 充电电流 | A |
| Max_Cell_Voltage | 最大单体电压 | V |
| Min_Cell_Voltage | 最小单体电压 | V |
| Max_Cell_Temperature | 最大单体温度 | °C |
| Min_Cell_Temperature | 最小单体温度 | °C |
| mileage | 里程数 | km |
| SOC | 充电状态（目标） | % |
| soh | 健康状态（目标） | 0-1 |

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据分析

```bash
python data_analysis.py
```

查看数据集的统计信息、分布情况等。

### 3. 训练模型

```bash
python train.py
```

训练过程会：
- 自动划分训练/验证/测试集
- 每个epoch后在验证集上评估
- 保存最佳模型到 `checkpoints/best_model.pth`
- 生成训练历史图表

**训练配置**（可在 `train.py` 中修改）:
```python
config = {
    'batch_size': 128,
    'sequence_length': 60,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 4,
    'model_type': 'standard'
}
```

### 4. 模型评估

```bash
python evaluate.py
```

评估脚本会生成：
- 评估报告（`results/evaluation_report.txt`）
- 预测散点图和误差分布（`results/predictions.png`）
- 时间序列预测对比（`results/time_series.png`）
- 误差分析图表（`results/error_analysis.png`）

### 5. 实时推理

```python
from predict import BatteryPredictor

# 创建预测器
predictor = BatteryPredictor()

# 准备输入数据（60个时间步）
charging_current = [-17.5] * 60
max_cell_voltage = [4.05] * 60
min_cell_voltage = [4.03] * 60
max_cell_temperature = [25.0] * 60
min_cell_temperature = [23.0] * 60
mileage = [120000.0] * 60

# 预测
result = predictor.predict(
    charging_current=charging_current,
    max_cell_voltage=max_cell_voltage,
    min_cell_voltage=min_cell_voltage,
    max_cell_temperature=max_cell_temperature,
    min_cell_temperature=min_cell_temperature,
    mileage=mileage
)

print(f"SOC: {result['SOC']:.2f}%")
print(f"SOH: {result['SOH']:.4f}")
```

或运行演示：
```bash
python predict.py
```

## 📁 项目结构

```
e:\SOH\
├── data/                           # 数据目录
│   └── battery_dataset_output_part_*.csv
├── venv/                           # 虚拟环境
├── checkpoints/                    # 模型检查点
│   ├── best_model.pth             # 最佳模型
│   └── latest_checkpoint.pth      # 最新检查点
├── results/                        # 评估结果
│   ├── evaluation_report.txt      # 评估报告
│   ├── predictions.png            # 预测图表
│   ├── time_series.png           # 时间序列图
│   └── error_analysis.png        # 误差分析图
├── data_analysis.py               # 数据分析脚本
├── dataset.py                     # 数据加载模块
├── model.py                       # 模型定义
├── train.py                       # 训练脚本
├── evaluate.py                    # 评估脚本
├── predict.py                     # 推理脚本
├── requirements.txt               # 依赖包
├── feature_scaler.pkl            # 特征标准化器
├── target_scaler.pkl             # 目标标准化器
└── README.md                      # 项目文档
```

## 🎯 模型性能指标

预期性能（在测试集上）：

| 指标 | SOC | SOH |
|------|-----|-----|
| MAE（平均绝对误差） | < 2% | < 0.02 |
| RMSE（均方根误差） | < 3% | < 0.03 |
| R²分数 | > 0.95 | > 0.90 |

*注：实际性能取决于训练轮数和数据质量*

## 🔧 高级配置

### 使用轻量级模型

如果计算资源有限，可以使用轻量级模型：

```python
# 在 train.py 中修改
config = {
    ...
    'model_type': 'lightweight',
    ...
}
```

轻量级模型参数量约为标准模型的1/4，但性能略有下降。

### 调整序列长度

修改 `sequence_length` 可以改变输入的时间窗口：

```python
# 30个时间步 = 5分钟历史数据
sequence_length = 30

# 120个时间步 = 20分钟历史数据
sequence_length = 120
```

## 📈 训练监控

训练过程中会实时显示：
- 训练损失和验证损失
- SOC和SOH的平均绝对误差(MAE)
- 学习率变化

训练完成后会自动生成训练历史曲线图。

## 💡 使用建议

1. **数据预处理**: 确保输入数据的质量和一致性
2. **序列长度**: 根据应用场景调整，建议10-20分钟的历史数据
3. **批次大小**: 根据显存调整，GPU建议128-256，CPU建议32-64
4. **学习率**: 初始1e-4，可根据训练曲线调整
5. **早停策略**: 如果验证损失不再下降，可提前停止训练

## 🐛 常见问题

### Q: CUDA out of memory
A: 减小 `batch_size` 或使用轻量级模型

### Q: 训练速度慢
A:
- 增加 `num_workers`（数据加载线程数）
- 使用GPU训练
- 减少数据文件数量（快速测试）

### Q: 预测结果不准确
A:
- 检查输入数据的格式和范围
- 确保使用了正确的标准化器
- 训练更多epoch
- 增加模型复杂度

## 📝 TODO

- [ ] 添加注意力可视化
- [ ] 实现在线学习/增量学习
- [ ] 支持多电池并行预测
- [ ] 添加异常检测功能
- [ ] 模型量化和加速
- [ ] Web API接口

## 📄 许可证

MIT License

## 👥 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题，请通过GitHub Issues联系。

---

**注意**: 本项目仅供学习和研究使用，实际应用前请充分测试和验证。

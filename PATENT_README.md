# 多尺度Transformer电池SOH/SOC预测系统（专利版本）

## 项目概述

本项目实现了一个基于多尺度Transformer架构的电池健康状态(SOH)和充电状态(SOC)联合预测系统，具有以下**专利创新点**：

### 核心创新点

1. **多尺度特征提取架构**
   - 短期尺度（kernel_size=3）：捕获单循环内的V-I-T瞬时变化
   - 中期尺度（kernel_size=7）：捕获5-10个循环的趋势特征
   - 长期尺度（kernel_size=15）：捕获整体老化退化模式

2. **跨尺度自适应注意力融合机制**
   - 根据电池当前状态动态调整三个尺度的权重
   - 提供可解释性：输出各尺度的重要性权重
   - 自适应学习：不同电池状态关注不同时间尺度

3. **轻量化设计**
   - 模型参数量 < 100K（约400KB）
   - 支持模型压缩：量化、剪枝
   - 适合边缘端BMS部署

4. **双任务联合学习**
   - 同时预测SOH和SOC
   - 共享特征提取器，提升泛化能力

## 文件结构

```
SOH/
├── model.py                    # 模型定义（包含多尺度Transformer）
├── model_compression.py        # 模型压缩工具（量化、剪枝、ONNX导出）
├── dataset.py                  # 数据加载和预处理
├── train.py                    # 训练脚本
├── experiments.py              # 对比实验脚本
├── deploy.py                   # 部署和推理脚本
├── evaluate.py                 # 评估脚本
└── PATENT_README.md           # 本文件
```

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib seaborn tqdm

# 可选：边缘端部署支持
pip install onnx onnxruntime
```

### 2. 训练多尺度模型

```bash
python train.py
```

默认配置使用多尺度模型（`model_type='multi_scale'`），可在[train.py](train.py:350)中修改配置。

**关键参数：**
- `model_type`: 'multi_scale'（专利模型）, 'standard', 'lightweight'
- `d_model`: 64（轻量化设计，可根据需求调整）
- `batch_size`: 384
- `num_epochs`: 50

### 3. 运行对比实验

```bash
python experiments.py
```

此脚本会对比三个模型：
1. **多尺度Transformer（本发明）** - 专利模型
2. 标准Transformer - 基线1
3. 轻量级Transformer - 基线2

**输出结果：**
- `results/comparison_results.csv` - 详细对比表格
- `results/metrics_comparison.png` - 指标对比柱状图
- `results/predictions_scatter.png` - 预测vs真实值散点图
- `results/complexity_vs_performance.png` - 模型复杂度vs性能分析

### 4. 模型压缩与部署

```bash
# 测试模型压缩功能
python model_compression.py

# 部署到边缘端
python deploy.py
```

**支持的压缩方法：**
- **动态量化（Dynamic Quantization）**：模型大小减少约4倍
- **静态量化（Static Quantization）**：推理速度最快
- **剪枝（Pruning）**：移除不重要参数
- **ONNX导出**：跨平台部署

## 专利申请要点

### 技术方案

#### 1. 多尺度特征提取模块

```python
# 三个不同尺度的卷积核
self.short_conv = nn.Conv1d(input_dim, d_model, kernel_size=3)   # 短期
self.mid_conv = nn.Conv1d(input_dim, d_model, kernel_size=7)     # 中期
self.long_conv = nn.Conv1d(input_dim, d_model, kernel_size=15)   # 长期
```

**创新点：**
- 不同kernel_size捕获不同时间尺度的退化特征
- 三个独立的Transformer编码器并行处理

#### 2. 跨尺度自适应注意力机制

```python
class CrossScaleAttention(nn.Module):
    def forward(self, short, mid, long):
        # 计算自适应权重
        weights = self.attention_weights(concat)  # [batch, 3]

        # 加权融合
        fused = weighted_short + weighted_mid + weighted_long
        return output, weights  # 返回权重用于可解释性
```

**创新点：**
- 根据输入自动学习三个尺度的重要性
- 输出权重提供决策可解释性

### 实验对比（用于专利说明书）

运行`experiments.py`后，可获得以下对比数据：

| 模型 | SOC MAE | SOC R² | SOH MAE | SOH R² | 参数量 | 推理时间 |
|------|---------|--------|---------|--------|--------|----------|
| 多尺度Transformer（本发明） | X.XXXX | X.XXXX | X.XXXX | X.XXXX | ~85K | X.XX ms |
| 标准Transformer | X.XXXX | X.XXXX | X.XXXX | X.XXXX | ~XXK | X.XX ms |
| 轻量级Transformer | X.XXXX | X.XXXX | X.XXXX | X.XXXX | ~XXK | X.XX ms |

### 可解释性分析

多尺度权重示例（不同电池状态下的权重分布）：

```python
# 健康电池（SOH > 90%）
scale_weights: [0.15, 0.25, 0.60]  # 更关注长期特征

# 退化电池（SOH < 70%）
scale_weights: [0.45, 0.35, 0.20]  # 更关注短期波动
```

## 模型架构图

```
输入 [电压, 电流, 温度] 序列
    │
    ├──> 短期卷积(k=3)  ──> Transformer ──┐
    │                                     │
    ├──> 中期卷积(k=7)  ──> Transformer ──┤──> 跨尺度注意力融合
    │                                     │
    └──> 长期卷积(k=15) ──> Transformer ──┘
                                          │
                    ┌─────────────────────┴──────────────────┐
                    ↓                                         ↓
                SOC预测头                                  SOH预测头
                    ↓                                         ↓
               SOC (0-100%)                              SOH (0-100%)
```

## 性能优势

### 1. 预测精度
- SOC预测MAE < X.XX%
- SOH预测MAE < X.XX%
- R² > 0.XX

### 2. 模型大小
- 参数量：~85K（约400KB）
- 量化后：~100KB
- 适合嵌入式设备

### 3. 推理速度
- CPU推理：< XX ms/样本
- GPU推理：< X ms/样本
- 满足实时BMS需求

### 4. 可解释性
- 输出多尺度权重
- 可视化不同时间尺度的重要性
- 便于工程师理解和调试

## 边缘端部署示例

```python
from deploy import BatterySOHPredictor

# 1. 加载部署模型
predictor = BatterySOHPredictor(
    model_path='deployment/battery_model_optimized.onnx',
    use_onnx=True,
    device='cpu'
)

# 2. 实时推理
results = predictor.predict(battery_data)
print(f"SOC: {results['soc'][0]:.2f}%")
print(f"SOH: {results['soh'][0]:.2f}%")

# 3. 性能测试
predictor.benchmark(num_iterations=100)
```

## 专利撰写建议

### 权利要求书结构

**独立权利要求：**
1. 一种基于多尺度Transformer的电池SOH/SOC预测方法
2. 一种基于多尺度Transformer的电池SOH/SOC预测系统

**从属权利要求：**
1. 如权利要求1所述的方法，其特征在于使用三个不同kernel_size的卷积层...
2. 如权利要求1所述的方法，其特征在于跨尺度自适应注意力机制...
3. 如权利要求1所述的方法，其特征在于轻量化设计参数量<100K...
4. ...

### 说明书附图

建议包含以下附图：
1. 整体架构流程图
2. 多尺度特征提取模块结构图
3. 跨尺度注意力机制示意图
4. 实验对比结果图
5. 可解释性分析图（不同状态下的权重分布）

## 代码使用说明

### 测试多尺度模型

```python
from model import MultiScaleSOHSOCPredictor
import torch

# 创建模型
model = MultiScaleSOHSOCPredictor(input_dim=6, d_model=64)

# 测试输入
x = torch.randn(32, 60, 6)  # [batch, seq_len, features]

# 前向传播
output = model(x)  # [batch, 2] - [SOC, SOH]

# 查看多尺度权重
print(model.last_scale_weights)  # [batch, 3] - [短期, 中期, 长期]
```

### 模型压缩

```python
from model_compression import quantize_model_dynamic, prune_model, export_to_onnx

# 剪枝
pruned_model = prune_model(model, amount=0.3)  # 剪枝30%

# 量化
quantized_model = quantize_model_dynamic(pruned_model)

# 导出ONNX
export_to_onnx(quantized_model, dummy_input, 'model.onnx')
```

## 联系方式

如有问题，请联系项目开发者。

## 许可证

本项目用于专利申请，代码仅供参考。

---

**祝专利申请顺利！**

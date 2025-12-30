# 快速开始 - GPU训练

## 1. 激活GPU环境

打开**Anaconda Prompt**（或普通命令行），执行：

```bash
conda activate battery_gpu
```

你应该看到命令提示符变成：
```
(battery_gpu) E:\SOH>
```

## 2. 安装PyTorch GPU版本

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

这会下载约2GB的文件，需要2-5分钟。

## 3. 安装其他依赖

```bash
pip install pandas scikit-learn matplotlib seaborn tqdm
```

## 4. 验证GPU

```bash
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**期望输出**：
```
CUDA可用: True
GPU: NVIDIA GeForce RTX 4060
```

如果看到这个输出，说明GPU配置成功！✅

## 5. 进入项目目录

```bash
cd E:\SOH
```

## 6. 测试模块

```bash
python test_modules.py
```

应该显示所有测试通过。

## 7. 开始训练

```bash
python train.py
```

训练会自动使用GPU。你应该看到：
- 设备显示为 `cuda`
- GPU利用率接近100%（可用`nvidia-smi`查看）

## 监控GPU使用

在另一个命令行窗口运行：

```bash
nvidia-smi -l 1
```

这会每秒刷新一次GPU使用情况。

## 预期训练时间

- **每个epoch**: 约2-5分钟
- **50个epochs**: 约2-4小时
- **显存占用**: 约2-4GB

## 常见问题

### Q: 显示 CUDA可用: False

**可能原因**：
1. NVIDIA驱动未安装或版本过旧
2. PyTorch安装的是CPU版本

**解决方法**：
```bash
# 检查NVIDIA驱动
nvidia-smi

# 重新安装GPU版本
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Q: CUDA out of memory

**解决方法**：
编辑 `train.py`，找到配置部分，修改：
```python
config = {
    'batch_size': 64,  # 从128改为64
    ...
}
```

### Q: 训练很慢，GPU使用率低

**解决方法**：
编辑 `train.py`，修改：
```python
config = {
    'batch_size': 256,  # 增大批次
    'num_workers': 8,   # 增加数据加载线程
    ...
}
```

## 完整训练流程

```bash
# 1. 激活环境
conda activate battery_gpu

# 2. 进入项目目录
cd E:\SOH

# 3. 开始训练（会训练50个epochs）
python train.py

# 4. 训练完成后评估
python evaluate.py

# 5. 测试推理
python predict.py
```

## 训练输出示例

```
配置:
  batch_size: 128
  sequence_length: 60
  num_epochs: 50
  learning_rate: 0.0001
  weight_decay: 1e-05
  num_workers: 4
  model_type: standard
  device: cuda

创建数据加载器...
找到 100 个数据文件
训练集: 70 文件
验证集: 15 文件
测试集: 15 文件

创建模型...
开始训练，共 50 个epoch
设备: cuda
模型参数量: 926,082

Epoch 1/50
------------------------------------------------------------
训练: 100%|██████████| 8000/8000 [03:45<00:00, 35.5it/s]
训练 - Loss: 0.1234, SOC MAE: 2.34, SOH MAE: 0.0123
验证: 100%|██████████| 1500/1500 [00:32<00:00, 46.2it/s]
验证 - Loss: 0.1156, SOC MAE: 2.12, SOH MAE: 0.0115
保存最佳模型到 checkpoints/best_model.pth

...
```

## 下一步

训练完成后查看结果：

1. **训练历史图**: `checkpoints/training_history.png`
2. **评估报告**: `results/evaluation_report.txt`
3. **预测可视化**: `results/predictions.png`

---

**祝训练顺利！如有问题，随时询问。** 🚀

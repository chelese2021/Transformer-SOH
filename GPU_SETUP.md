# RTX 4060 GPU配置指南

## 问题诊断

当前环境使用的是**Python 3.14**，但PyTorch的预编译CUDA版本目前只支持到**Python 3.12**。

因此，当前环境只能使用**CPU版本**的PyTorch。

## 解决方案

### 方案1：使用Python 3.12重新创建环境（强烈推荐）

#### 步骤1：安装Python 3.12

从Python官网下载并安装Python 3.12：
https://www.python.org/downloads/release/python-3120/

选择：Windows installer (64-bit)

#### 步骤2：创建新的虚拟环境

```bash
# 在项目目录中
cd e:\SOH

# 使用Python 3.12创建虚拟环境
python3.12 -m venv venv_gpu

# 或者如果Python 3.12是默认版本
py -3.12 -m venv venv_gpu
```

#### 步骤3：激活虚拟环境

```bash
# Windows
venv_gpu\Scripts\activate
```

#### 步骤4：安装依赖

```bash
# 升级pip
python -m pip install --upgrade pip

# 安装GPU版本的PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或者如果CUDA 12.1有问题，使用CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
```

#### 步骤5：验证GPU

```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

应该看到：
```
CUDA可用: True
GPU: NVIDIA GeForce RTX 4060
```

### 方案2：使用Conda（推荐，更简单）

Conda可以自动管理Python版本和CUDA依赖。

#### 步骤1：安装Miniconda

下载并安装：https://docs.conda.io/en/latest/miniconda.html

#### 步骤2：创建环境

```bash
# 创建Python 3.12环境并安装PyTorch GPU版本
conda create -n battery_soc python=3.12 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 激活环境
conda activate battery_soc

# 安装其他依赖
pip install pandas scikit-learn matplotlib seaborn tqdm
```

#### 步骤3：验证

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 方案3：继续使用当前CPU环境（不推荐）

如果你暂时不想重新配置环境，可以先用CPU版本：

**优点**：
- 无需重新配置
- 可以测试代码功能

**缺点**：
- 训练速度慢约10-50倍
- 不适合大规模训练

**使用CPU训练的建议**：
1. 减少数据量（只使用部分CSV文件）
2. 减小批次大小（batch_size=32或16）
3. 减少训练轮数（num_epochs=10）
4. 使用轻量级模型

修改[train.py](train.py:53-61)中的配置：
```python
config = {
    'batch_size': 32,  # 减小
    'sequence_length': 60,
    'num_epochs': 10,  # 减少
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 0,  # CPU下设为0
    'model_type': 'lightweight',  # 使用轻量级模型
    'device': 'cpu'  # 明确指定CPU
}
```

修改[dataset.py](dataset.py:114-115)中的数据加载：
```python
# 只使用前5个文件进行快速测试
train_files = all_files[:5]
val_files = all_files[5:7]
test_files = all_files[7:10]
```

## RTX 4060 训练性能预估

使用GPU后的性能提升：

| 配置项 | CPU | RTX 4060 | 加速比 |
|--------|-----|----------|--------|
| 每个epoch时间 | ~30-60分钟 | ~2-5分钟 | ~10-15x |
| 总训练时间(50 epochs) | ~25-50小时 | ~2-4小时 | ~10-15x |
| 批次大小 | 32-64 | 128-256 | 2-4x |
| 推理速度 | ~50-100 samples/s | ~1000+ samples/s | ~10-20x |

## GPU训练配置建议

在GPU环境中，使用以下配置获得最佳性能：

```python
# train.py 配置
config = {
    'batch_size': 256,          # GPU可以使用更大的批次
    'sequence_length': 60,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 4,           # 数据加载并行
    'model_type': 'standard',   # 使用标准模型
    'device': 'cuda'            # 使用GPU
}
```

## 监控GPU使用

训练时可以打开另一个命令行窗口，运行：

```bash
# 安装nvidia-smi（通常随NVIDIA驱动自动安装）
nvidia-smi

# 实时监控（每秒刷新）
nvidia-smi -l 1
```

或者在代码中添加GPU内存监控：

```python
import torch
print(f"GPU显存已使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU显存已缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

## 常见问题

### Q1: CUDA out of memory
**解决方法**：
- 减小batch_size（如256→128→64）
- 使用混合精度训练（fp16）
- 减小模型尺寸（使用lightweight模型）

### Q2: GPU利用率低
**解决方法**：
- 增加batch_size
- 增加num_workers（数据加载线程）
- 使用pin_memory=True

### Q3: 训练不稳定
**解决方法**：
- 降低学习率
- 使用梯度裁剪（已在train.py中实现）
- 增加warmup阶段

## 下一步

配置好GPU环境后：

1. 运行测试验证GPU可用：
   ```bash
   python test_modules.py
   ```

2. 开始GPU训练：
   ```bash
   python train.py
   ```

3. 监控训练过程，检查GPU使用率

4. 训练完成后评估模型：
   ```bash
   python evaluate.py
   ```

---

**推荐：使用方案1或方案2配置Python 3.12环境，以充分利用RTX 4060的性能！**

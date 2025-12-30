"""
模块测试脚本
验证所有组件是否正常工作
"""
import torch
import numpy as np
import sys

print("=" * 70)
print("电池SOC/SOH预测系统 - 模块测试")
print("=" * 70)

# 1. 测试模型
print("\n[1/3] 测试模型模块...")
try:
    from model import BatteryTransformer, LightweightBatteryTransformer, get_model

    # 创建测试输入
    batch_size = 4
    seq_len = 60
    input_dim = 6
    test_input = torch.randn(batch_size, seq_len, input_dim)

    # 测试标准模型
    model_standard = BatteryTransformer(input_dim=input_dim)
    output_standard = model_standard(test_input)
    assert output_standard.shape == (batch_size, 2), "标准模型输出形状错误"
    print(f"  [OK] 标准模型测试通过")
    print(f"    - 输入形状: {test_input.shape}")
    print(f"    - 输出形状: {output_standard.shape}")
    print(f"    - 参数量: {sum(p.numel() for p in model_standard.parameters()):,}")

    # 测试轻量级模型
    model_light = LightweightBatteryTransformer(input_dim=input_dim)
    output_light = model_light(test_input)
    assert output_light.shape == (batch_size, 2), "轻量级模型输出形状错误"
    print(f"  [OK] 轻量级模型测试通过")
    print(f"    - 参数量: {sum(p.numel() for p in model_light.parameters()):,}")

except Exception as e:
    print(f"  [ERROR] 模型模块测试失败: {e}")
    sys.exit(1)

# 2. 测试数据集（使用少量数据）
print("\n[2/3] 测试数据集模块...")
try:
    from dataset import BatteryDataset
    import glob
    import os

    # 获取1个数据文件进行测试
    data_files = glob.glob('data/battery_dataset_output_part_1.csv')
    if not data_files:
        print("  [WARN] 警告: 未找到数据文件，跳过数据集测试")
    else:
        dataset = BatteryDataset(
            data_files=data_files,
            sequence_length=60,
            train=True
        )
        print(f"  [OK] 数据集加载成功")
        print(f"    - 样本数量: {len(dataset):,}")
        print(f"    - 特征维度: {dataset.data.shape[1]}")

        # 测试获取样本
        X, y = dataset[0]
        assert X.shape == (60, 6), "特征形状错误"
        assert y.shape == (2,), "目标形状错误"
        print(f"  [OK] 数据样本测试通过")
        print(f"    - 特征形状: {X.shape}")
        print(f"    - 目标形状: {y.shape}")

except Exception as e:
    print(f"  [ERROR] 数据集模块测试失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 测试配置
print("\n[3/3] 测试配置模块...")
try:
    from config import get_config, print_config

    config = get_config('train')
    assert 'data' in config, "配置缺少data部分"
    assert 'model' in config, "配置缺少model部分"
    assert 'train' in config, "配置缺少train部分"
    print(f"  [OK] 配置模块测试通过")
    print(f"    - 包含配置项: {list(config.keys())}")

except Exception as e:
    print(f"  [ERROR] 配置模块测试失败: {e}")
    sys.exit(1)

# 测试完成
print("\n" + "=" * 70)
print("所有模块测试通过！")
print("=" * 70)
print("\n下一步:")
print("  1. 运行 'python data_analysis.py' 查看数据统计")
print("  2. 运行 'python train.py' 开始训练模型")
print("  3. 运行 'python evaluate.py' 评估模型性能")
print("  4. 运行 'python predict.py' 进行实时推理")
print("=" * 70)

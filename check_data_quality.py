"""
检查数据质量问题
"""
import numpy as np
import pandas as pd
import pickle
import os


def check_data_quality():
    """检查训练集、验证集、测试集是否有重叠或异常"""

    print("=" * 100)
    print("数据质量检查")
    print("=" * 100)

    # 加载标准化器
    with open('target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)

    # 读取一些原始数据文件检查
    data_dir = 'data'
    files = sorted([f for f in os.listdir(data_dir) if f.startswith('battery_dataset_output_part_')])

    print(f"\n找到 {len(files)} 个数据文件")

    # 读取前几个文件检查SOH分布
    all_soh = []
    all_soc = []

    print("\n读取前5个文件检查数据分布...")
    for i, file in enumerate(files[:5]):
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)

        print(f"\n文件 {file}:")
        print(f"  样本数: {len(df)}")
        print(f"  SOH唯一值数量: {df['soh'].nunique()}")
        print(f"  SOH值范围: [{df['soh'].min():.6f}, {df['soh'].max():.6f}]")
        print(f"  SOH平均值: {df['soh'].mean():.6f}")
        print(f"  SOH标准差: {df['soh'].std():.6f}")

        # 检查是否有很多重复的SOH值
        soh_counts = df['soh'].value_counts()
        print(f"  最常见的SOH值: {soh_counts.index[0]:.6f} (出现 {soh_counts.values[0]} 次)")

        # 检查SOC
        print(f"  SOC唯一值数量: {df['SOC'].nunique()}")
        print(f"  SOC值范围: [{df['SOC'].min():.4f}, {df['SOC'].max():.4f}]")

        all_soh.extend(df['soh'].values)
        all_soc.extend(df['SOC'].values)

    # 整体统计
    all_soh = np.array(all_soh)
    all_soc = np.array(all_soc)

    print("\n" + "=" * 100)
    print("前5个文件的整体统计")
    print("=" * 100)
    print(f"\nSOH统计:")
    print(f"  总样本数: {len(all_soh)}")
    print(f"  唯一SOH值数量: {len(np.unique(all_soh))}")
    print(f"  SOH值范围: [{all_soh.min():.6f}, {all_soh.max():.6f}]")
    print(f"  SOH平均值: {all_soh.mean():.6f}")
    print(f"  SOH标准差: {all_soh.std():.6f}")

    # 检查SOH是否集中在某些值
    unique_soh, counts = np.unique(all_soh, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]

    print(f"\n最常出现的10个SOH值:")
    print(f"{'SOH值':<15} {'出现次数':<15} {'占比':<15}")
    print("-" * 45)
    for i in range(min(10, len(unique_soh))):
        idx = sorted_indices[i]
        print(f"{unique_soh[idx]:<15.6f} {counts[idx]:<15} {counts[idx]/len(all_soh)*100:<15.2f}%")

    print(f"\nSOC统计:")
    print(f"  唯一SOC值数量: {len(np.unique(all_soc))}")
    print(f"  SOC值范围: [{all_soc.min():.4f}, {all_soc.max():.4f}]")
    print(f"  SOC平均值: {all_soc.mean():.4f}")
    print(f"  SOC标准差: {all_soc.std():.4f}")

    # 检查特定的异常SOH值
    print("\n" + "=" * 100)
    print("异常值检查")
    print("=" * 100)

    # 检查0.582776这个值（在预测最差样本中频繁出现）
    suspicious_soh = 0.582776
    count_suspicious = np.sum(np.abs(all_soh - suspicious_soh) < 0.0001)
    print(f"\nSOH = {suspicious_soh} 的样本数: {count_suspicious} ({count_suspicious/len(all_soh)*100:.2f}%)")

    # 检查数据是否是时间序列
    print("\n" + "=" * 100)
    print("时间序列特性检查")
    print("=" * 100)

    # 读取第一个文件检查时间戳
    first_file = os.path.join(data_dir, files[0])
    df_first = pd.read_csv(first_file)

    print(f"\n数据列: {list(df_first.columns)}")

    if 'time' in df_first.columns or 'timestamp' in df_first.columns:
        print("数据包含时间戳列")
    else:
        print("数据不包含时间戳列")

    # 检查SOH是否随时间变化
    print(f"\n检查前1000行SOH的变化:")
    sample_soh = df_first['soh'].head(1000).values
    print(f"  第一个值: {sample_soh[0]:.6f}")
    print(f"  最后一个值: {sample_soh[-1]:.6f}")
    print(f"  变化量: {sample_soh[-1] - sample_soh[0]:.6f}")
    print(f"  是否恒定: {np.all(sample_soh == sample_soh[0])}")

    # 检查是否有连续相同的SOH值
    consecutive_same = 0
    max_consecutive = 0
    current_consecutive = 1

    for i in range(1, len(sample_soh)):
        if abs(sample_soh[i] - sample_soh[i-1]) < 1e-8:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1

    print(f"  最长连续相同SOH值的长度: {max_consecutive}")

    print("=" * 100)


if __name__ == '__main__':
    check_data_quality()

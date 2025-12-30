import pandas as pd
import numpy as np
import glob
import os

# 读取第一个文件查看数据结构
file_path = 'data/battery_dataset_output_part_1.csv'
df_sample = pd.read_csv(file_path)

print("=" * 60)
print("数据集结构分析")
print("=" * 60)
print(f"\n数据集列名：\n{df_sample.columns.tolist()}")
print(f"\n数据形状：{df_sample.shape}")
print(f"\n数据类型：\n{df_sample.dtypes}")

print("\n" + "=" * 60)
print("数据统计信息")
print("=" * 60)
print(df_sample.describe())

print("\n" + "=" * 60)
print("缺失值检查")
print("=" * 60)
print(df_sample.isnull().sum())

print("\n" + "=" * 60)
print("SOC和SOH范围")
print("=" * 60)
print(f"SOC范围: [{df_sample['SOC'].min():.2f}, {df_sample['SOC'].max():.2f}]")
print(f"SOH范围: [{df_sample['soh'].min():.4f}, {df_sample['soh'].max():.4f}]")

print("\n" + "=" * 60)
print("前5行样本数据")
print("=" * 60)
print(df_sample.head())

print("\n" + "=" * 60)
print("所有CSV文件统计")
print("=" * 60)
csv_files = sorted(glob.glob('data/battery_dataset_output_part_*.csv'))
print(f"CSV文件总数: {len(csv_files)}")
print(f"单个文件平均行数: {len(df_sample):,}")
print(f"预估总数据量: {len(csv_files) * len(df_sample):,} 行")

# 计算时间序列长度
print("\n" + "=" * 60)
print("时间序列分析")
print("=" * 60)
print(f"时间戳范围: {df_sample['Timestamp'].min()} - {df_sample['Timestamp'].max()}")
print(f"时间采样间隔: {df_sample['Timestamp'].diff().mode().values[0]} 秒")

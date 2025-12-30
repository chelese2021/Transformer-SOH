"""
检查数据的组织结构，了解文件是如何组织的
"""
import pandas as pd
import numpy as np
import os
import glob


def inspect_data_structure():
    """检查数据结构"""
    data_dir = 'data'
    files = sorted(glob.glob(os.path.join(data_dir, 'battery_dataset_output_part_*.csv')))

    print("=" * 100)
    print("数据结构分析")
    print("=" * 100)

    # 检查前10个文件
    print(f"\n总文件数: {len(files)}")
    print(f"\n检查前10个文件的结构:\n")

    file_info = []

    for i, file in enumerate(files[:10]):
        df = pd.read_csv(file)

        info = {
            'file': os.path.basename(file),
            'rows': len(df),
            'unique_soh': df['soh'].nunique(),
            'soh_min': df['soh'].min(),
            'soh_max': df['soh'].max(),
            'soh_mean': df['soh'].mean(),
            'soh_std': df['soh'].std(),
            'has_timestamp': 'Timestamp' in df.columns
        }

        file_info.append(info)

        print(f"文件 {i+1}: {os.path.basename(file)}")
        print(f"  样本数: {info['rows']}")
        print(f"  SOH唯一值: {info['unique_soh']}")
        print(f"  SOH范围: [{info['soh_min']:.6f}, {info['soh_max']:.6f}]")
        print(f"  SOH均值±std: {info['soh_mean']:.6f} ± {info['soh_std']:.6f}")

        # 检查时间戳
        if 'Timestamp' in df.columns:
            print(f"  时间戳范围: {df['Timestamp'].min()} -> {df['Timestamp'].max()}")
            # 检查时间是否连续
            time_diff = df['Timestamp'].diff().dropna()
            print(f"  时间间隔: 平均 {time_diff.mean():.2f}s, 中位数 {time_diff.median():.2f}s")

        # 检查里程
        if 'mileage' in df.columns:
            print(f"  里程范围: {df['mileage'].min():.0f} -> {df['mileage'].max():.0f} km")

        # 检查SOH是否在文件内变化
        soh_first = df['soh'].iloc[0]
        soh_last = df['soh'].iloc[-1]
        soh_change = soh_last - soh_first
        print(f"  SOH变化: {soh_first:.6f} -> {soh_last:.6f} (Δ={soh_change:.6f})")

        print()

    # 检查文件之间的连续性
    print("\n" + "=" * 100)
    print("检查文件间连续性")
    print("=" * 100)

    print("\n比较相邻文件的边界值:\n")

    for i in range(min(5, len(files)-1)):
        df1 = pd.read_csv(files[i])
        df2 = pd.read_csv(files[i+1])

        print(f"文件 {i+1} 结尾 -> 文件 {i+2} 开头:")
        print(f"  SOH: {df1['soh'].iloc[-1]:.6f} -> {df2['soh'].iloc[0]:.6f}")
        print(f"  SOC: {df1['SOC'].iloc[-1]:.4f} -> {df2['SOC'].iloc[0]:.4f}")

        if 'Timestamp' in df1.columns:
            time_gap = df2['Timestamp'].iloc[0] - df1['Timestamp'].iloc[-1]
            print(f"  时间间隔: {time_gap:.2f}s")

        if 'mileage' in df1.columns:
            print(f"  里程: {df1['mileage'].iloc[-1]:.0f} -> {df2['mileage'].iloc[0]:.0f} km")

        print()

    # 检查是否每个文件代表不同的电池
    print("\n" + "=" * 100)
    print("假设检验：每个文件是否代表不同的电池或时间段？")
    print("=" * 100)

    # 如果文件间SOH差异很大，可能是不同电池
    # 如果文件间SOH连续，可能是同一电池的不同时间段

    soh_gaps = []
    for i in range(min(20, len(files)-1)):
        df1 = pd.read_csv(files[i])
        df2 = pd.read_csv(files[i+1])
        gap = abs(df2['soh'].iloc[0] - df1['soh'].iloc[-1])
        soh_gaps.append(gap)

    soh_gaps = np.array(soh_gaps)

    print(f"\n前20对相邻文件的SOH间隔:")
    print(f"  最小间隔: {soh_gaps.min():.6f}")
    print(f"  最大间隔: {soh_gaps.max():.6f}")
    print(f"  平均间隔: {soh_gaps.mean():.6f}")
    print(f"  中位数: {np.median(soh_gaps):.6f}")

    # 如果间隔很小(<0.01)，说明可能是连续的；如果间隔大，可能是不同电池
    large_gaps = np.sum(soh_gaps > 0.01)
    print(f"\n大间隔(>0.01)的数量: {large_gaps}/{len(soh_gaps)}")

    if soh_gaps.mean() < 0.01:
        print("\n推断: 文件间SOH变化很小，可能是同一批电池的连续记录")
        print("       或者是按时间顺序组织的数据")
    else:
        print("\n推断: 文件间SOH变化较大，可能每个文件代表不同的电池")

    print("\n" + "=" * 100)


if __name__ == '__main__':
    inspect_data_structure()

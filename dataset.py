"""
电池数据集加载和预处理模块
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob
import os
from typing import Tuple, List
import pickle


class BatteryDataset(Dataset):
    """
    电池时间序列数据集

    参数：
        data_files: CSV文件路径列表
        sequence_length: 时间序列窗口长度
        prediction_horizon: 预测未来多少步
        train: 是否为训练集
        scaler: 数据标准化器
    """

    def __init__(
        self,
        data_files: List[str],
        sequence_length: int = 60,  # 10分钟的数据 (60 * 10秒)
        prediction_horizon: int = 1,
        train: bool = True,
        scaler: StandardScaler = None
    ):
        self.data_files = data_files
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.train = train

        # 特征列（输入）
        self.feature_cols = [
            'Charging_Current',
            'Max_Cell_Voltage',
            'Min_Cell_Voltage',
            'Max_Cell_Temperature',
            'Min_Cell_Temperature',
            'mileage'
        ]

        # 目标列（输出）
        self.target_cols = ['SOC', 'soh']

        # 加载数据
        print(f"加载 {len(data_files)} 个文件...")
        self.data, self.targets = self._load_data()

        # 数据标准化
        if scaler is None:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)

        # 目标值标准化器
        if train:
            self.target_scaler = StandardScaler()
            self.targets = self.target_scaler.fit_transform(self.targets)
        else:
            # 验证集和测试集暂时不标准化，稍后会使用训练集的scaler
            self.target_scaler = None

        print(f"数据形状: {self.data.shape}, 目标形状: {self.targets.shape}")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载并合并所有CSV文件"""
        all_features = []
        all_targets = []

        for file_path in self.data_files:
            df = pd.read_csv(file_path)

            # 提取特征和目标
            features = df[self.feature_cols].values
            targets = df[self.target_cols].values

            all_features.append(features)
            all_targets.append(targets)

        # 合并所有数据
        features = np.vstack(all_features)
        targets = np.vstack(all_targets)

        return features, targets

    def __len__(self) -> int:
        """返回可生成的样本数量"""
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回一个时间序列样本

        返回：
            X: 输入序列 [sequence_length, num_features]
            y: 目标值 [2] (SOC, SOH)
        """
        # 输入序列
        X = self.data[idx:idx + self.sequence_length]

        # 预测目标（未来的SOC和SOH）
        target_idx = idx + self.sequence_length + self.prediction_horizon - 1
        y = self.targets[target_idx]

        return torch.FloatTensor(X), torch.FloatTensor(y)


def create_dataloaders(
    data_dir: str = 'data',
    batch_size: int = 64,
    sequence_length: int = 60,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    shuffle_files: bool = False,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """
    创建训练、验证和测试数据加载器

    参数：
        data_dir: 数据目录
        batch_size: 批次大小
        sequence_length: 序列长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        num_workers: 数据加载线程数
        shuffle_files: 是否随机打乱文件顺序（默认False，保持时间序列顺序）
        random_seed: 随机种子

    返回：
        train_loader, val_loader, test_loader, feature_scaler, target_scaler
    """
    # 获取所有CSV文件
    all_files = sorted(glob.glob(os.path.join(data_dir, 'battery_dataset_output_part_*.csv')))
    print(f"找到 {len(all_files)} 个数据文件")

    # 可选：随机打乱文件顺序
    if shuffle_files:
        np.random.seed(random_seed)
        np.random.shuffle(all_files)
        print(f"已使用随机种子 {random_seed} 打乱文件顺序")

    # 划分数据集
    n_files = len(all_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)

    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]

    print(f"训练集: {len(train_files)} 文件")
    print(f"验证集: {len(val_files)} 文件")
    print(f"测试集: {len(test_files)} 文件")

    # 创建数据集
    train_dataset = BatteryDataset(
        train_files,
        sequence_length=sequence_length,
        train=True
    )

    val_dataset = BatteryDataset(
        val_files,
        sequence_length=sequence_length,
        train=False,
        scaler=train_dataset.scaler
    )
    val_dataset.target_scaler = train_dataset.target_scaler
    # 使用训练集的scaler标准化验证集目标值
    val_dataset.targets = val_dataset.target_scaler.transform(val_dataset.targets)

    test_dataset = BatteryDataset(
        test_files,
        sequence_length=sequence_length,
        train=False,
        scaler=train_dataset.scaler
    )
    test_dataset.target_scaler = train_dataset.target_scaler
    # 使用训练集的scaler标准化测试集目标值
    test_dataset.targets = test_dataset.target_scaler.transform(test_dataset.targets)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 保存标准化器
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    with open('target_scaler.pkl', 'wb') as f:
        pickle.dump(train_dataset.target_scaler, f)

    return train_loader, val_loader, test_loader, train_dataset.scaler, train_dataset.target_scaler


if __name__ == '__main__':
    # 测试数据加载
    train_loader, val_loader, test_loader, _, _ = create_dataloaders(
        batch_size=32,
        sequence_length=60,
        num_workers=0
    )

    print("\n数据加载器测试:")
    for X, y in train_loader:
        print(f"输入形状: {X.shape}")  # [batch_size, sequence_length, num_features]
        print(f"目标形状: {y.shape}")  # [batch_size, 2]
        print(f"输入范围: [{X.min():.3f}, {X.max():.3f}]")
        print(f"目标范围: [{y.min():.3f}, {y.max():.3f}]")
        break

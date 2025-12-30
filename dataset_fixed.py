"""
修复后的电池数据集加载模块
解决时间序列数据泄露问题
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


class BatteryDatasetFixed(Dataset):
    """
    修复后的电池时间序列数据集

    改进：
    1. 不允许滑动窗口跨越文件边界
    2. 每个文件作为独立的时间序列单元
    3. 正确处理文件间的独立性
    """

    def __init__(
        self,
        data_files: List[str],
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        train: bool = True,
        feature_scaler: StandardScaler = None,
        target_scaler: StandardScaler = None
    ):
        self.data_files = data_files
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.train = train

        # 特征列
        self.feature_cols = [
            'Charging_Current',
            'Max_Cell_Voltage',
            'Min_Cell_Voltage',
            'Max_Cell_Temperature',
            'Min_Cell_Temperature',
            'mileage'
        ]

        # 目标列
        self.target_cols = ['SOC', 'soh']

        # 加载数据（每个文件独立保存）
        print(f"加载 {len(data_files)} 个文件...")
        self.file_data = self._load_data_by_file()

        # 构建样本索引（关键：不跨文件边界）
        self.sample_indices = self._build_sample_indices()
        print(f"生成 {len(self.sample_indices)} 个有效样本")

        # 数据标准化
        if feature_scaler is None:
            self.feature_scaler = StandardScaler()
            self._fit_scaler()
        else:
            self.feature_scaler = feature_scaler

        if train and target_scaler is None:
            self.target_scaler = StandardScaler()
            self._fit_target_scaler()
        else:
            self.target_scaler = target_scaler

        # 应用标准化
        self._apply_scaling()

    def _load_data_by_file(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        按文件加载数据，每个文件独立保存

        返回：
            List[(features, targets)] 每个文件的特征和目标
        """
        file_data = []

        for file_path in self.data_files:
            df = pd.read_csv(file_path)

            features = df[self.feature_cols].values
            targets = df[self.target_cols].values

            file_data.append((features, targets))

        return file_data

    def _build_sample_indices(self) -> List[Tuple[int, int]]:
        """
        构建样本索引，确保不跨文件边界

        返回：
            List[(file_idx, row_idx)] 样本索引列表
        """
        sample_indices = []

        for file_idx, (features, targets) in enumerate(self.file_data):
            n_rows = len(features)
            # 每个文件内可以生成的样本数
            max_start_idx = n_rows - self.sequence_length - self.prediction_horizon + 1

            for row_idx in range(max_start_idx):
                sample_indices.append((file_idx, row_idx))

        return sample_indices

    def _fit_scaler(self):
        """拟合特征标准化器"""
        # 收集所有训练数据
        all_features = []
        for features, _ in self.file_data:
            all_features.append(features)

        all_features = np.vstack(all_features)
        self.feature_scaler.fit(all_features)

    def _fit_target_scaler(self):
        """拟合目标标准化器"""
        # 收集所有训练目标
        all_targets = []
        for _, targets in self.file_data:
            all_targets.append(targets)

        all_targets = np.vstack(all_targets)
        self.target_scaler.fit(all_targets)

    def _apply_scaling(self):
        """应用标准化"""
        scaled_data = []

        for features, targets in self.file_data:
            # 标准化特征
            features_scaled = self.feature_scaler.transform(features)

            # 标准化目标
            if self.target_scaler is not None:
                targets_scaled = self.target_scaler.transform(targets)
            else:
                targets_scaled = targets

            scaled_data.append((features_scaled, targets_scaled))

        self.file_data = scaled_data

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本（不会跨文件边界）
        """
        file_idx, row_idx = self.sample_indices[idx]
        features, targets = self.file_data[file_idx]

        # 输入序列
        X = features[row_idx:row_idx + self.sequence_length]

        # 预测目标
        target_idx = row_idx + self.sequence_length + self.prediction_horizon - 1
        y = targets[target_idx]

        return torch.FloatTensor(X), torch.FloatTensor(y)


def create_dataloaders_fixed(
    data_dir: str = 'data',
    batch_size: int = 64,
    sequence_length: int = 60,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    random_seed: int = 42,
    shuffle_files: bool = True  # 默认打乱文件
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """
    创建修复后的数据加载器

    改进：
    1. 强制打乱文件顺序（避免时间序列偏差）
    2. 按文件划分数据集（确保独立性）
    3. 不允许跨文件边界的样本
    """
    # 获取所有文件
    all_files = sorted(glob.glob(os.path.join(data_dir, 'battery_dataset_output_part_*.csv')))
    print(f"找到 {len(all_files)} 个数据文件")

    # 打乱文件顺序（重要！）
    if shuffle_files:
        np.random.seed(random_seed)
        np.random.shuffle(all_files)
        print(f"已使用随机种子 {random_seed} 打乱文件顺序")

    # 按文件划分数据集
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
    print("\n创建训练集...")
    train_dataset = BatteryDatasetFixed(
        train_files,
        sequence_length=sequence_length,
        train=True
    )

    print("\n创建验证集...")
    val_dataset = BatteryDatasetFixed(
        val_files,
        sequence_length=sequence_length,
        train=False,
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler
    )

    print("\n创建测试集...")
    test_dataset = BatteryDatasetFixed(
        test_files,
        sequence_length=sequence_length,
        train=False,
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler
    )

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
    with open('feature_scaler_fixed.pkl', 'wb') as f:
        pickle.dump(train_dataset.feature_scaler, f)
    with open('target_scaler_fixed.pkl', 'wb') as f:
        pickle.dump(train_dataset.target_scaler, f)

    print(f"\n数据加载完成！")
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    print(f"测试样本: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, train_dataset.feature_scaler, train_dataset.target_scaler


# 测试数据加载器
if __name__ == '__main__':
    print("=" * 100)
    print("测试修复后的数据加载器")
    print("=" * 100)

    train_loader, val_loader, test_loader, _, _ = create_dataloaders_fixed(
        batch_size=256,
        sequence_length=60,
        num_workers=0,  # Windows兼容性
        shuffle_files=True,
        random_seed=42
    )

    print("\n" + "=" * 100)
    print("数据集统计")
    print("=" * 100)

    # 测试加载一个批次
    X_batch, y_batch = next(iter(train_loader))
    print(f"\n批次形状:")
    print(f"  输入: {X_batch.shape}  # [batch_size, sequence_length, num_features]")
    print(f"  目标: {y_batch.shape}  # [batch_size, 2]")

    print("\n修复后的数据加载器测试完成！")

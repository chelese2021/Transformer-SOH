"""
模型评估和可视化脚本
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from tqdm import tqdm
from typing import Tuple, Dict

from model import get_model
from dataset import create_dataloaders


class Evaluator:
    """
    模型评估器
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader,
        target_scaler,
        device: str = 'cuda',
        save_dir: str = 'results'
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.target_scaler = target_scaler
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        评估模型

        返回：
            y_true: 真实值 [N, 2]
            y_pred: 预测值 [N, 2]
        """
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for X, y in tqdm(self.test_loader, desc='评估'):
                X = X.to(self.device)

                # 预测
                outputs = self.model(X)

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        # 合并所有批次
        y_pred = np.vstack(all_predictions)
        y_true = np.vstack(all_targets)

        # 反标准化
        y_pred = self.target_scaler.inverse_transform(y_pred)
        y_true = self.target_scaler.inverse_transform(y_true)

        return y_true, y_pred

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算评估指标"""
        metrics = {}

        # SOC指标
        soc_true = y_true[:, 0]
        soc_pred = y_pred[:, 0]
        metrics['soc_mae'] = mean_absolute_error(soc_true, soc_pred)
        metrics['soc_rmse'] = np.sqrt(mean_squared_error(soc_true, soc_pred))
        metrics['soc_r2'] = r2_score(soc_true, soc_pred)
        metrics['soc_mape'] = np.mean(np.abs((soc_true - soc_pred) / (soc_true + 1e-8))) * 100

        # SOH指标
        soh_true = y_true[:, 1]
        soh_pred = y_pred[:, 1]
        metrics['soh_mae'] = mean_absolute_error(soh_true, soh_pred)
        metrics['soh_rmse'] = np.sqrt(mean_squared_error(soh_true, soh_pred))
        metrics['soh_r2'] = r2_score(soh_true, soh_pred)
        metrics['soh_mape'] = np.mean(np.abs((soh_true - soh_pred) / (soh_true + 1e-8))) * 100

        return metrics

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray):
        """绘制预测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # SOC散点图
        axes[0, 0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.3, s=1)
        axes[0, 0].plot([y_true[:, 0].min(), y_true[:, 0].max()],
                        [y_true[:, 0].min(), y_true[:, 0].max()],
                        'r--', linewidth=2, label='理想预测')
        axes[0, 0].set_xlabel('真实SOC (%)', fontsize=12)
        axes[0, 0].set_ylabel('预测SOC (%)', fontsize=12)
        axes[0, 0].set_title('SOC预测散点图', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # SOH散点图
        axes[0, 1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.3, s=1)
        axes[0, 1].plot([y_true[:, 1].min(), y_true[:, 1].max()],
                        [y_true[:, 1].min(), y_true[:, 1].max()],
                        'r--', linewidth=2, label='理想预测')
        axes[0, 1].set_xlabel('真实SOH', fontsize=12)
        axes[0, 1].set_ylabel('预测SOH', fontsize=12)
        axes[0, 1].set_title('SOH预测散点图', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # SOC误差分布
        soc_error = y_pred[:, 0] - y_true[:, 0]
        axes[1, 0].hist(soc_error, bins=100, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('预测误差 (%)', fontsize=12)
        axes[1, 0].set_ylabel('频数', fontsize=12)
        axes[1, 0].set_title(f'SOC误差分布 (μ={soc_error.mean():.3f}, σ={soc_error.std():.3f})',
                            fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # SOH误差分布
        soh_error = y_pred[:, 1] - y_true[:, 1]
        axes[1, 1].hist(soh_error, bins=100, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('预测误差', fontsize=12)
        axes[1, 1].set_ylabel('频数', fontsize=12)
        axes[1, 1].set_title(f'SOH误差分布 (μ={soh_error.mean():.4f}, σ={soh_error.std():.4f})',
                            fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'predictions.png'), dpi=300, bbox_inches='tight')
        print(f"预测结果图保存到 {os.path.join(self.save_dir, 'predictions.png')}")
        plt.close()

    def plot_time_series(self, y_true: np.ndarray, y_pred: np.ndarray, num_samples: int = 1000):
        """绘制时间序列预测"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # 只显示部分样本
        indices = np.arange(min(num_samples, len(y_true)))

        # SOC时间序列
        axes[0].plot(indices, y_true[indices, 0], label='真实SOC', alpha=0.7, linewidth=1.5)
        axes[0].plot(indices, y_pred[indices, 0], label='预测SOC', alpha=0.7, linewidth=1.5)
        axes[0].fill_between(indices,
                             y_true[indices, 0],
                             y_pred[indices, 0],
                             alpha=0.3, label='误差区域')
        axes[0].set_xlabel('样本索引', fontsize=12)
        axes[0].set_ylabel('SOC (%)', fontsize=12)
        axes[0].set_title('SOC时间序列预测', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # SOH时间序列
        axes[1].plot(indices, y_true[indices, 1], label='真实SOH', alpha=0.7, linewidth=1.5)
        axes[1].plot(indices, y_pred[indices, 1], label='预测SOH', alpha=0.7, linewidth=1.5)
        axes[1].fill_between(indices,
                             y_true[indices, 1],
                             y_pred[indices, 1],
                             alpha=0.3, label='误差区域')
        axes[1].set_xlabel('样本索引', fontsize=12)
        axes[1].set_ylabel('SOH', fontsize=12)
        axes[1].set_title('SOH时间序列预测', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'time_series.png'), dpi=300, bbox_inches='tight')
        print(f"时间序列图保存到 {os.path.join(self.save_dir, 'time_series.png')}")
        plt.close()

    def plot_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray):
        """绘制误差分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # SOC误差 vs 真实值
        soc_error = np.abs(y_pred[:, 0] - y_true[:, 0])
        axes[0, 0].scatter(y_true[:, 0], soc_error, alpha=0.3, s=1)
        axes[0, 0].set_xlabel('真实SOC (%)', fontsize=12)
        axes[0, 0].set_ylabel('绝对误差 (%)', fontsize=12)
        axes[0, 0].set_title('SOC误差 vs 真实值', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # SOH误差 vs 真实值
        soh_error = np.abs(y_pred[:, 1] - y_true[:, 1])
        axes[0, 1].scatter(y_true[:, 1], soh_error, alpha=0.3, s=1)
        axes[0, 1].set_xlabel('真实SOH', fontsize=12)
        axes[0, 1].set_ylabel('绝对误差', fontsize=12)
        axes[0, 1].set_title('SOH误差 vs 真实值', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # SOC误差箱型图（按区间）
        soc_bins = np.linspace(y_true[:, 0].min(), y_true[:, 0].max(), 11)
        soc_bin_indices = np.digitize(y_true[:, 0], soc_bins)
        soc_errors_by_bin = [soc_error[soc_bin_indices == i] for i in range(1, len(soc_bins))]
        axes[1, 0].boxplot(soc_errors_by_bin, labels=[f'{int(soc_bins[i])}' for i in range(len(soc_bins)-1)])
        axes[1, 0].set_xlabel('SOC区间 (%)', fontsize=12)
        axes[1, 0].set_ylabel('绝对误差 (%)', fontsize=12)
        axes[1, 0].set_title('不同SOC区间的误差分布', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # SOH误差箱型图（按区间）
        soh_bins = np.linspace(y_true[:, 1].min(), y_true[:, 1].max(), 11)
        soh_bin_indices = np.digitize(y_true[:, 1], soh_bins)
        soh_errors_by_bin = [soh_error[soh_bin_indices == i] for i in range(1, len(soh_bins))]
        axes[1, 1].boxplot(soh_errors_by_bin, labels=[f'{soh_bins[i]:.2f}' for i in range(len(soh_bins)-1)])
        axes[1, 1].set_xlabel('SOH区间', fontsize=12)
        axes[1, 1].set_ylabel('绝对误差', fontsize=12)
        axes[1, 1].set_title('不同SOH区间的误差分布', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"误差分析图保存到 {os.path.join(self.save_dir, 'error_analysis.png')}")
        plt.close()

    def generate_report(self, metrics: Dict):
        """生成评估报告"""
        report_path = os.path.join(self.save_dir, 'evaluation_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("电池SOC和SOH预测模型 - 评估报告\n")
            f.write("=" * 70 + "\n\n")

            f.write("SOC预测性能:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  平均绝对误差(MAE):     {metrics['soc_mae']:.4f} %\n")
            f.write(f"  均方根误差(RMSE):      {metrics['soc_rmse']:.4f} %\n")
            f.write(f"  R²分数:                {metrics['soc_r2']:.6f}\n")
            f.write(f"  平均绝对百分比误差:    {metrics['soc_mape']:.4f} %\n\n")

            f.write("SOH预测性能:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  平均绝对误差(MAE):     {metrics['soh_mae']:.6f}\n")
            f.write(f"  均方根误差(RMSE):      {metrics['soh_rmse']:.6f}\n")
            f.write(f"  R²分数:                {metrics['soh_r2']:.6f}\n")
            f.write(f"  平均绝对百分比误差:    {metrics['soh_mape']:.4f} %\n\n")

            f.write("=" * 70 + "\n")

        print(f"\n评估报告保存到 {report_path}")

        # 在控制台打印
        with open(report_path, 'r', encoding='utf-8') as f:
            print(f.read())


def main():
    """主函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建数据加载器
    print("\n创建数据加载器...")
    _, _, test_loader, _, target_scaler = create_dataloaders(
        batch_size=256,
        sequence_length=60,
        num_workers=4
    )

    # 加载模型（需要与训练时配置一致）
    print("\n加载模型...")
    model = get_model(
        model_type='lightweight',  # 使用轻量级模型
        input_dim=6,
        d_model=64,  # 轻量级模型参数
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )

    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型: {checkpoint_path}")
        print(f"训练epoch: {checkpoint['epoch']}")
    else:
        print(f"警告: 未找到检查点 {checkpoint_path}")
        print("请先训练模型！")
        return

    # 创建评估器
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        target_scaler=target_scaler,
        device=device
    )

    # 评估
    print("\n开始评估...")
    y_true, y_pred = evaluator.evaluate()

    # 计算指标
    print("\n计算评估指标...")
    metrics = evaluator.compute_metrics(y_true, y_pred)

    # 生成报告
    evaluator.generate_report(metrics)

    # 可视化
    print("\n生成可视化图表...")
    evaluator.plot_predictions(y_true, y_pred)
    evaluator.plot_time_series(y_true, y_pred, num_samples=2000)
    evaluator.plot_error_analysis(y_true, y_pred)

    print("\n评估完成！")


if __name__ == '__main__':
    main()

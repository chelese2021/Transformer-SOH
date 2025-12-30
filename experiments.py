"""
对比实验模块
用于专利论文的实验对比：多尺度模型 vs 基线模型
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import time
import os

from model import get_model, MultiScaleSOHSOCPredictor, BatteryTransformer, LightweightBatteryTransformer
from dataset import create_dataloaders


class ModelEvaluator:
    """
    模型评估器
    用于对比不同模型的性能
    """

    def __init__(self, test_loader, target_scaler, device='cuda'):
        self.test_loader = test_loader
        self.target_scaler = target_scaler
        self.device = device

    def evaluate_model(self, model: nn.Module, model_name: str) -> Dict:
        """
        评估单个模型

        返回：
            metrics: 包含各项指标的字典
        """
        model.eval()
        model = model.to(self.device)

        soc_preds = []
        soc_trues = []
        soh_preds = []
        soh_trues = []

        inference_times = []

        print(f"\n评估模型: {model_name}")
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)

                # 测量推理时间
                start_time = time.time()
                outputs = model(X)
                inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
                inference_times.append(inference_time)

                # 收集预测结果
                outputs = outputs.cpu().numpy()
                y = y.numpy()

                # 反标准化
                outputs = self.target_scaler.inverse_transform(outputs)
                y = self.target_scaler.inverse_transform(y)

                soc_preds.extend(outputs[:, 0])
                soc_trues.extend(y[:, 0])
                soh_preds.extend(outputs[:, 1])
                soh_trues.extend(y[:, 1])

        # 计算指标
        metrics = {
            'model_name': model_name,
            # SOC指标
            'SOC_MAE': mean_absolute_error(soc_trues, soc_preds),
            'SOC_RMSE': np.sqrt(mean_squared_error(soc_trues, soc_preds)),
            'SOC_R2': r2_score(soc_trues, soc_preds),
            'SOC_MAPE': np.mean(np.abs((np.array(soc_trues) - np.array(soc_preds)) / np.array(soc_trues))) * 100,
            # SOH指标
            'SOH_MAE': mean_absolute_error(soh_trues, soh_preds),
            'SOH_RMSE': np.sqrt(mean_squared_error(soh_trues, soh_preds)),
            'SOH_R2': r2_score(soh_trues, soh_preds),
            'SOH_MAPE': np.mean(np.abs((np.array(soh_trues) - np.array(soh_preds)) / np.array(soh_trues))) * 100,
            # 推理性能
            'avg_inference_time_ms': np.mean(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            # 模型复杂度
            'parameters': sum(p.numel() for p in model.parameters()),
            # 原始预测值（用于可视化）
            'soc_preds': soc_preds,
            'soc_trues': soc_trues,
            'soh_preds': soh_preds,
            'soh_trues': soh_trues
        }

        return metrics

    def compare_models(self, models_dict: Dict[str, nn.Module]) -> pd.DataFrame:
        """
        对比多个模型

        参数：
            models_dict: {模型名称: 模型实例}

        返回：
            results_df: 对比结果DataFrame
        """
        all_metrics = []

        for model_name, model in models_dict.items():
            metrics = self.evaluate_model(model, model_name)
            all_metrics.append(metrics)

        # 创建对比表
        comparison_data = []
        for metrics in all_metrics:
            comparison_data.append({
                '模型': metrics['model_name'],
                'SOC MAE': f"{metrics['SOC_MAE']:.4f}",
                'SOC RMSE': f"{metrics['SOC_RMSE']:.4f}",
                'SOC R²': f"{metrics['SOC_R2']:.4f}",
                'SOC MAPE(%)': f"{metrics['SOC_MAPE']:.2f}",
                'SOH MAE': f"{metrics['SOH_MAE']:.4f}",
                'SOH RMSE': f"{metrics['SOH_RMSE']:.4f}",
                'SOH R²': f"{metrics['SOH_R2']:.4f}",
                'SOH MAPE(%)': f"{metrics['SOH_MAPE']:.2f}",
                '推理时间(ms)': f"{metrics['avg_inference_time_ms']:.2f}",
                '参数量': f"{metrics['parameters']:,}"
            })

        results_df = pd.DataFrame(comparison_data)

        return results_df, all_metrics

    def plot_comparison(self, all_metrics: List[Dict], save_dir='results'):
        """
        绘制对比图表

        参数：
            all_metrics: 所有模型的评估指标
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. 指标对比柱状图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        metrics_to_plot = [
            ('SOC_MAE', 'SOC MAE (越低越好)'),
            ('SOC_RMSE', 'SOC RMSE (越低越好)'),
            ('SOC_R2', 'SOC R² (越高越好)'),
            ('SOH_MAE', 'SOH MAE (越低越好)'),
            ('SOH_RMSE', 'SOH RMSE (越低越好)'),
            ('SOH_R2', 'SOH R² (越高越好)')
        ]

        for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            model_names = [m['model_name'] for m in all_metrics]
            values = [m[metric_key] for m in all_metrics]

            bars = ax.bar(range(len(model_names)), values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=15, ha='right')
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_ylabel('值')
            ax.grid(axis='y', alpha=0.3)

            # 在柱子上显示数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"✓ 指标对比图保存到: {os.path.join(save_dir, 'metrics_comparison.png')}")

        # 2. 预测vs真实值散点图
        fig, axes = plt.subplots(len(all_metrics), 2, figsize=(12, 5*len(all_metrics)))
        if len(all_metrics) == 1:
            axes = axes.reshape(1, -1)

        for idx, metrics in enumerate(all_metrics):
            # SOC散点图
            ax = axes[idx, 0]
            ax.scatter(metrics['soc_trues'], metrics['soc_preds'], alpha=0.5, s=10)
            ax.plot([min(metrics['soc_trues']), max(metrics['soc_trues'])],
                   [min(metrics['soc_trues']), max(metrics['soc_trues'])],
                   'r--', lw=2, label='理想预测')
            ax.set_xlabel('真实SOC (%)')
            ax.set_ylabel('预测SOC (%)')
            ax.set_title(f"{metrics['model_name']} - SOC预测\nR²={metrics['SOC_R2']:.4f}, MAE={metrics['SOC_MAE']:.4f}")
            ax.legend()
            ax.grid(alpha=0.3)

            # SOH散点图
            ax = axes[idx, 1]
            ax.scatter(metrics['soh_trues'], metrics['soh_preds'], alpha=0.5, s=10, color='orange')
            ax.plot([min(metrics['soh_trues']), max(metrics['soh_trues'])],
                   [min(metrics['soh_trues']), max(metrics['soh_trues'])],
                   'r--', lw=2, label='理想预测')
            ax.set_xlabel('真实SOH (%)')
            ax.set_ylabel('预测SOH (%)')
            ax.set_title(f"{metrics['model_name']} - SOH预测\nR²={metrics['SOH_R2']:.4f}, MAE={metrics['SOH_MAE']:.4f}")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'predictions_scatter.png'), dpi=300, bbox_inches='tight')
        print(f"✓ 预测散点图保存到: {os.path.join(save_dir, 'predictions_scatter.png')}")

        # 3. 模型复杂度vs性能
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        params = [m['parameters'] / 1000 for m in all_metrics]  # 转换为K
        soc_mae = [m['SOC_MAE'] for m in all_metrics]
        soh_mae = [m['SOH_MAE'] for m in all_metrics]
        model_names = [m['model_name'] for m in all_metrics]

        # SOC MAE vs 参数量
        axes[0].scatter(params, soc_mae, s=200, alpha=0.7, c=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        for i, name in enumerate(model_names):
            axes[0].annotate(name, (params[i], soc_mae[i]),
                           fontsize=10, ha='center', va='bottom')
        axes[0].set_xlabel('模型参数量 (K)', fontsize=12)
        axes[0].set_ylabel('SOC MAE', fontsize=12)
        axes[0].set_title('模型复杂度 vs SOC性能', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)

        # SOH MAE vs 参数量
        axes[1].scatter(params, soh_mae, s=200, alpha=0.7, c=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        for i, name in enumerate(model_names):
            axes[1].annotate(name, (params[i], soh_mae[i]),
                           fontsize=10, ha='center', va='bottom')
        axes[1].set_xlabel('模型参数量 (K)', fontsize=12)
        axes[1].set_ylabel('SOH MAE', fontsize=12)
        axes[1].set_title('模型复杂度 vs SOH性能', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'complexity_vs_performance.png'), dpi=300, bbox_inches='tight')
        print(f"✓ 复杂度vs性能图保存到: {os.path.join(save_dir, 'complexity_vs_performance.png')}")

        plt.close('all')


def run_experiments():
    """
    运行完整的对比实验
    """
    print("="*80)
    print("开始对比实验")
    print("="*80)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n加载测试数据...")
    _, _, test_loader, feature_scaler, target_scaler = create_dataloaders(
        batch_size=128,
        sequence_length=60,
        num_workers=0
    )

    # 创建评估器
    evaluator = ModelEvaluator(test_loader, target_scaler, device)

    # 定义要对比的模型
    models_dict = {}

    # 1. 多尺度模型（专利模型）
    print("\n加载多尺度模型...")
    multi_scale_model = MultiScaleSOHSOCPredictor(input_dim=6, d_model=64)
    if os.path.exists('checkpoints/best_model.pth'):
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        multi_scale_model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ 已加载训练好的多尺度模型")
    else:
        print("⚠ 未找到训练好的模型，使用随机初始化（建议先训练模型）")
    models_dict['多尺度Transformer（本发明）'] = multi_scale_model

    # 2. 标准Transformer（基线1）
    print("\n创建标准Transformer基线...")
    standard_model = BatteryTransformer(
        input_dim=6,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.1
    )
    models_dict['标准Transformer'] = standard_model

    # 3. 轻量级Transformer（基线2）
    print("\n创建轻量级Transformer基线...")
    lightweight_model = LightweightBatteryTransformer(
        input_dim=6,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    models_dict['轻量级Transformer'] = lightweight_model

    # 运行对比实验
    print("\n"+"="*80)
    print("开始评估模型...")
    print("="*80)

    results_df, all_metrics = evaluator.compare_models(models_dict)

    # 打印结果
    print("\n"+"="*80)
    print("实验结果对比")
    print("="*80)
    print(results_df.to_string(index=False))

    # 保存结果
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/comparison_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存到: results/comparison_results.csv")

    # 绘制对比图
    print("\n生成对比图表...")
    evaluator.plot_comparison(all_metrics, save_dir='results')

    # 分析结果
    print("\n"+"="*80)
    print("性能分析")
    print("="*80)

    # 找出最佳模型
    best_soc_idx = np.argmin([m['SOC_MAE'] for m in all_metrics])
    best_soh_idx = np.argmin([m['SOH_MAE'] for m in all_metrics])
    fastest_idx = np.argmin([m['avg_inference_time_ms'] for m in all_metrics])
    smallest_idx = np.argmin([m['parameters'] for m in all_metrics])

    print(f"✓ SOC预测最佳: {all_metrics[best_soc_idx]['model_name']}")
    print(f"  MAE: {all_metrics[best_soc_idx]['SOC_MAE']:.4f}, R²: {all_metrics[best_soc_idx]['SOC_R2']:.4f}")

    print(f"\n✓ SOH预测最佳: {all_metrics[best_soh_idx]['model_name']}")
    print(f"  MAE: {all_metrics[best_soh_idx]['SOH_MAE']:.4f}, R²: {all_metrics[best_soh_idx]['SOH_R2']:.4f}")

    print(f"\n✓ 推理速度最快: {all_metrics[fastest_idx]['model_name']}")
    print(f"  推理时间: {all_metrics[fastest_idx]['avg_inference_time_ms']:.2f} ms/batch")

    print(f"\n✓ 模型最小: {all_metrics[smallest_idx]['model_name']}")
    print(f"  参数量: {all_metrics[smallest_idx]['parameters']:,}")

    print("\n"+"="*80)
    print("实验完成！")
    print("="*80)


if __name__ == '__main__':
    # 设置绘图样式
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示

    # 运行实验
    run_experiments()

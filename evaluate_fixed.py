"""
评估修复后的模型
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

from model import get_model
from dataset_fixed import create_dataloaders_fixed

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_model():
    """评估模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # 加载数据
    print("加载数据...")
    _, _, test_loader, _, target_scaler = create_dataloaders_fixed(
        batch_size=256,
        sequence_length=60,
        num_workers=0,
        shuffle_files=True,
        random_seed=42
    )

    # 加载模型
    print("\n加载模型...")
    checkpoint_path = 'checkpoints_fixed/best_model.pth'

    if not os.path.exists(checkpoint_path):
        print(f"错误: 未找到模型文件 {checkpoint_path}")
        print("请先运行 train_fixed.py 训练模型")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = get_model(
        model_type=config['model_type'],
        input_dim=6,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型训练了 {checkpoint['epoch']} 个epoch")
    print(f"最佳验证损失: {checkpoint['val_loss']:.6f}\n")

    # 预测
    print("在测试集上评估...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # 合并结果
    y_pred = np.vstack(all_predictions)
    y_true = np.vstack(all_targets)

    # 反标准化
    y_pred = target_scaler.inverse_transform(y_pred)
    y_true = target_scaler.inverse_transform(y_true)

    # 计算指标
    print("\n" + "=" * 100)
    print("评估结果")
    print("=" * 100)

    # SOC指标
    soc_true = y_true[:, 0]
    soc_pred = y_pred[:, 0]
    soc_mae = mean_absolute_error(soc_true, soc_pred)
    soc_rmse = np.sqrt(mean_squared_error(soc_true, soc_pred))
    soc_r2 = r2_score(soc_true, soc_pred)
    # 修复MAPE计算（避免除以接近0的值）
    soc_mape = np.mean(np.abs((soc_true - soc_pred) / np.clip(soc_true, 1.0, None))) * 100

    print("\nSOC预测性能:")
    print("-" * 100)
    print(f"  平均绝对误差 (MAE):    {soc_mae:.4f}%")
    print(f"  均方根误差 (RMSE):     {soc_rmse:.4f}%")
    print(f"  R2分数:                {soc_r2:.6f}")
    print(f"  平均绝对百分比误差:    {soc_mape:.4f}%")

    # SOH指标
    soh_true = y_true[:, 1]
    soh_pred = y_pred[:, 1]
    soh_mae = mean_absolute_error(soh_true, soh_pred)
    soh_rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))
    soh_r2 = r2_score(soh_true, soh_pred)
    soh_mape = np.mean(np.abs((soh_true - soh_pred) / soh_true)) * 100

    print("\nSOH预测性能:")
    print("-" * 100)
    print(f"  平均绝对误差 (MAE):    {soh_mae:.6f} ({soh_mae*100:.2f}%)")
    print(f"  均方根误差 (RMSE):     {soh_rmse:.6f} ({soh_rmse*100:.2f}%)")
    print(f"  R2分数:                {soh_r2:.6f}")
    print(f"  平均绝对百分比误差:    {soh_mape:.4f}%")
    print("=" * 100)

    # 显示10个随机样本
    print("\n随机10个测试样本的预测结果:")
    print("-" * 100)
    np.random.seed(42)
    random_indices = np.random.choice(len(y_true), 10, replace=False)

    print(f"{'样本ID':<10} {'真实SOC(%)':<12} {'预测SOC(%)':<12} {'SOC误差':<12} "
          f"{'真实SOH':<12} {'预测SOH':<12} {'SOH误差':<12}")
    print("-" * 100)

    for idx in random_indices:
        soc_err = soc_pred[idx] - soc_true[idx]
        soh_err = soh_pred[idx] - soh_true[idx]
        print(f"{idx:<10} {soc_true[idx]:<12.4f} {soc_pred[idx]:<12.4f} {soc_err:<+12.4f} "
              f"{soh_true[idx]:<12.6f} {soh_pred[idx]:<12.6f} {soh_err:<+12.6f}")

    # 保存报告
    os.makedirs('results_fixed', exist_ok=True)
    report_path = 'results_fixed/evaluation_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("修复后的模型 - 评估报告\n")
        f.write("=" * 100 + "\n\n")

        f.write("SOC预测性能:\n")
        f.write("-" * 100 + "\n")
        f.write(f"  平均绝对误差 (MAE):    {soc_mae:.4f}%\n")
        f.write(f"  均方根误差 (RMSE):     {soc_rmse:.4f}%\n")
        f.write(f"  R2分数:                {soc_r2:.6f}\n")
        f.write(f"  平均绝对百分比误差:    {soc_mape:.4f}%\n\n")

        f.write("SOH预测性能:\n")
        f.write("-" * 100 + "\n")
        f.write(f"  平均绝对误差 (MAE):    {soh_mae:.6f} ({soh_mae*100:.2f}%)\n")
        f.write(f"  均方根误差 (RMSE):     {soh_rmse:.6f} ({soh_rmse*100:.2f}%)\n")
        f.write(f"  R2分数:                {soh_r2:.6f}\n")
        f.write(f"  平均绝对百分比误差:    {soh_mape:.4f}%\n\n")

        f.write("=" * 100 + "\n")

    print(f"\n报告已保存到: {report_path}")

    # 绘制结果
    print("\n生成可视化图表...")
    plot_results(y_true, y_pred)

    print("\n评估完成！")


def plot_results(y_true, y_pred):
    """绘制评估结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # SOC散点图
    axes[0, 0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.1, s=1)
    axes[0, 0].plot([y_true[:, 0].min(), y_true[:, 0].max()],
                    [y_true[:, 0].min(), y_true[:, 0].max()],
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('True SOC (%)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted SOC (%)', fontsize=12)
    axes[0, 0].set_title('SOC Prediction', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # SOH散点图
    axes[0, 1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.1, s=1)
    axes[0, 1].plot([y_true[:, 1].min(), y_true[:, 1].max()],
                    [y_true[:, 1].min(), y_true[:, 1].max()],
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('True SOH', fontsize=12)
    axes[0, 1].set_ylabel('Predicted SOH', fontsize=12)
    axes[0, 1].set_title('SOH Prediction', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # SOC误差分布
    soc_error = y_pred[:, 0] - y_true[:, 0]
    axes[1, 0].hist(soc_error, bins=100, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Prediction Error (%)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title(f'SOC Error Distribution (μ={soc_error.mean():.3f}, σ={soc_error.std():.3f})',
                        fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # SOH误差分布
    soh_error = y_pred[:, 1] - y_true[:, 1]
    axes[1, 1].hist(soh_error, bins=100, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Prediction Error', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title(f'SOH Error Distribution (μ={soh_error.mean():.4f}, σ={soh_error.std():.4f})',
                        fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_fixed/predictions.png', dpi=300, bbox_inches='tight')
    print("图表已保存到: results_fixed/predictions.png")
    plt.close()


if __name__ == '__main__':
    evaluate_model()

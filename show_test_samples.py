"""
显示测试集中的具体样本预测结果
"""
import torch
import numpy as np
import pickle
import os
from model import get_model
from dataset import create_dataloaders


def show_test_samples(num_samples=10):
    """显示测试集中的样本预测结果"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # 创建数据加载器
    print("加载测试数据...")
    _, _, test_loader, _, target_scaler = create_dataloaders(
        batch_size=256,
        sequence_length=60,
        num_workers=4
    )

    # 加载模型
    print("加载模型...")
    model = get_model(
        model_type='lightweight',
        input_dim=6,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )

    checkpoint_path = 'checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"模型训练epoch: {checkpoint['epoch']}\n")

    # 获取第一批数据
    X_batch, y_batch = next(iter(test_loader))

    # 只取前num_samples个样本
    X_samples = X_batch[:num_samples].to(device)
    y_samples = y_batch[:num_samples].cpu().numpy()

    # 预测
    with torch.no_grad():
        predictions = model(X_samples).cpu().numpy()

    # 反标准化
    y_true = target_scaler.inverse_transform(y_samples)
    y_pred = target_scaler.inverse_transform(predictions)

    # 显示结果
    print("=" * 100)
    print("测试集样本预测结果 (前10个样本)")
    print("=" * 100)
    print(f"{'样本序号':<8} {'真实SOC(%)':<12} {'预测SOC(%)':<12} {'SOC误差(%)':<12} {'真实SOH':<12} {'预测SOH':<12} {'SOH误差':<12}")
    print("-" * 100)

    for i in range(num_samples):
        soc_true = y_true[i, 0]
        soc_pred = y_pred[i, 0]
        soc_error = soc_pred - soc_true

        soh_true = y_true[i, 1]
        soh_pred = y_pred[i, 1]
        soh_error = soh_pred - soh_true

        print(f"{i+1:<8} {soc_true:<12.4f} {soc_pred:<12.4f} {soc_error:<+12.4f} "
              f"{soh_true:<12.6f} {soh_pred:<12.6f} {soh_error:<+12.6f}")

    print("-" * 100)

    # 计算这10个样本的平均误差
    soc_mae = np.mean(np.abs(y_pred[:, 0] - y_true[:, 0]))
    soc_rmse = np.sqrt(np.mean((y_pred[:, 0] - y_true[:, 0])**2))

    soh_mae = np.mean(np.abs(y_pred[:, 1] - y_true[:, 1]))
    soh_rmse = np.sqrt(np.mean((y_pred[:, 1] - y_true[:, 1])**2))

    print(f"\n这10个样本的统计:")
    print(f"  SOC - MAE: {soc_mae:.4f}%,  RMSE: {soc_rmse:.4f}%")
    print(f"  SOH - MAE: {soh_mae:.6f},  RMSE: {soh_rmse:.6f}")
    print("=" * 100)


if __name__ == '__main__':
    show_test_samples(10)

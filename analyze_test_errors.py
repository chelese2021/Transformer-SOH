"""
分析测试集的误差分布，找出最好和最差的预测样本
"""
import torch
import numpy as np
import pickle
from model import get_model
from dataset import create_dataloaders


def analyze_errors():
    """分析测试集的误差分布"""
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
    print("加载模型...\n")
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 收集所有预测结果
    print("收集所有预测结果...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # 合并
    y_pred = np.vstack(all_predictions)
    y_true = np.vstack(all_targets)

    # 反标准化
    y_pred = target_scaler.inverse_transform(y_pred)
    y_true = target_scaler.inverse_transform(y_true)

    # 计算误差
    soc_errors = np.abs(y_pred[:, 0] - y_true[:, 0])
    soh_errors = np.abs(y_pred[:, 1] - y_true[:, 1])

    # 统计分析
    print("=" * 100)
    print("误差分布统计")
    print("=" * 100)

    print("\nSOC误差分布:")
    print(f"  最小误差: {soc_errors.min():.4f}%")
    print(f"  最大误差: {soc_errors.max():.4f}%")
    print(f"  平均误差: {soc_errors.mean():.4f}%")
    print(f"  中位数: {np.median(soc_errors):.4f}%")
    print(f"  标准差: {soc_errors.std():.4f}%")
    print(f"  95%分位数: {np.percentile(soc_errors, 95):.4f}%")
    print(f"  99%分位数: {np.percentile(soc_errors, 99):.4f}%")

    print("\nSOH误差分布:")
    print(f"  最小误差: {soh_errors.min():.6f}")
    print(f"  最大误差: {soh_errors.max():.6f}")
    print(f"  平均误差: {soh_errors.mean():.6f}")
    print(f"  中位数: {np.median(soh_errors):.6f}")
    print(f"  标准差: {soh_errors.std():.6f}")
    print(f"  95%分位数: {np.percentile(soh_errors, 95):.6f}")
    print(f"  99%分位数: {np.percentile(soh_errors, 99):.6f}")

    # 找出预测最好的10个样本
    print("\n" + "=" * 100)
    print("预测最好的10个样本")
    print("=" * 100)
    best_indices = np.argsort(soh_errors)[:10]
    print(f"{'样本ID':<10} {'真实SOC(%)':<12} {'预测SOC(%)':<12} {'SOC误差(%)':<12} {'真实SOH':<12} {'预测SOH':<12} {'SOH误差':<12}")
    print("-" * 100)
    for idx in best_indices:
        print(f"{idx:<10} {y_true[idx, 0]:<12.4f} {y_pred[idx, 0]:<12.4f} {soc_errors[idx]:<12.4f} "
              f"{y_true[idx, 1]:<12.6f} {y_pred[idx, 1]:<12.6f} {soh_errors[idx]:<12.6f}")

    # 找出预测最差的10个样本
    print("\n" + "=" * 100)
    print("预测最差的10个样本 (SOH误差最大)")
    print("=" * 100)
    worst_indices = np.argsort(soh_errors)[-10:][::-1]
    print(f"{'样本ID':<10} {'真实SOC(%)':<12} {'预测SOC(%)':<12} {'SOC误差(%)':<12} {'真实SOH':<12} {'预测SOH':<12} {'SOH误差':<12}")
    print("-" * 100)
    for idx in worst_indices:
        print(f"{idx:<10} {y_true[idx, 0]:<12.4f} {y_pred[idx, 0]:<12.4f} {soc_errors[idx]:<12.4f} "
              f"{y_true[idx, 1]:<12.6f} {y_pred[idx, 1]:<12.6f} {soh_errors[idx]:<12.6f}")

    # 随机采样10个样本
    print("\n" + "=" * 100)
    print("随机采样10个样本")
    print("=" * 100)
    np.random.seed(42)
    random_indices = np.random.choice(len(y_true), 10, replace=False)
    print(f"{'样本ID':<10} {'真实SOC(%)':<12} {'预测SOC(%)':<12} {'SOC误差(%)':<12} {'真实SOH':<12} {'预测SOH':<12} {'SOH误差':<12}")
    print("-" * 100)
    for idx in random_indices:
        print(f"{idx:<10} {y_true[idx, 0]:<12.4f} {y_pred[idx, 0]:<12.4f} {soc_errors[idx]:<12.4f} "
              f"{y_true[idx, 1]:<12.6f} {y_pred[idx, 1]:<12.6f} {soh_errors[idx]:<12.6f}")

    # 检查SOH值的分布
    print("\n" + "=" * 100)
    print("数据分布检查")
    print("=" * 100)
    print("\nSOH真实值分布:")
    print(f"  最小值: {y_true[:, 1].min():.6f}")
    print(f"  最大值: {y_true[:, 1].max():.6f}")
    print(f"  平均值: {y_true[:, 1].mean():.6f}")
    print(f"  标准差: {y_true[:, 1].std():.6f}")
    print(f"  范围: {y_true[:, 1].max() - y_true[:, 1].min():.6f}")

    print("\nSOC真实值分布:")
    print(f"  最小值: {y_true[:, 0].min():.4f}%")
    print(f"  最大值: {y_true[:, 0].max():.4f}%")
    print(f"  平均值: {y_true[:, 0].mean():.4f}%")
    print(f"  标准差: {y_true[:, 0].std():.4f}%")

    # 检查有多少SOC接近0
    near_zero_soc = np.sum(np.abs(y_true[:, 0]) < 1.0)
    print(f"\n接近0的SOC值 (<1%): {near_zero_soc} 个 ({near_zero_soc/len(y_true)*100:.2f}%)")

    print("=" * 100)


if __name__ == '__main__':
    analyze_errors()

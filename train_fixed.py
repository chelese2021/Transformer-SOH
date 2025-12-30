"""
使用修复后的数据加载器重新训练模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os

from model import get_model
from dataset_fixed import create_dataloaders_fixed


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc='训练')

    for X, y in progress_bar:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)

        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for X, y in tqdm(val_loader, desc='验证'):
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    """主训练函数"""
    # 设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # 超参数
    config = {
        'batch_size': 256,
        'sequence_length': 60,
        'learning_rate': 1e-4,
        'epochs': 20,
        'model_type': 'lightweight',  # 或 'standard'
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1,
    }

    print("=" * 100)
    print("配置参数")
    print("=" * 100)
    for key, value in config.items():
        print(f"{key}: {value}")
    print()

    # 创建数据加载器（使用修复版本）
    print("=" * 100)
    print("加载数据（修复版）")
    print("=" * 100)
    train_loader, val_loader, test_loader, _, _ = create_dataloaders_fixed(
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        num_workers=0,  # Windows兼容性
        shuffle_files=True,  # 重要：打乱文件顺序
        random_seed=42
    )

    # 创建模型
    print("\n" + "=" * 100)
    print("创建模型")
    print("=" * 100)
    model = get_model(
        model_type=config['model_type'],
        input_dim=6,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 训练循环
    print("\n" + "=" * 100)
    print("开始训练")
    print("=" * 100)

    best_val_loss = float('inf')
    os.makedirs('checkpoints_fixed', exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 100)

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # 调整学习率
        scheduler.step(val_loss)

        print(f"\n训练损失: {train_loss:.6f}")
        print(f"验证损失: {val_loss:.6f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': config
            }, 'checkpoints_fixed/best_model.pth')
            print(f"✓ 保存最佳模型 (验证损失: {val_loss:.6f})")

        # 保存最新检查点
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'config': config
        }, 'checkpoints_fixed/latest_checkpoint.pth')

    print("\n" + "=" * 100)
    print("训练完成！")
    print("=" * 100)
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型保存在: checkpoints_fixed/best_model.pth")

    # 保存训练历史
    np.save('checkpoints_fixed/train_losses.npy', np.array(train_losses))
    np.save('checkpoints_fixed/val_losses.npy', np.array(val_losses))


if __name__ == '__main__':
    main()

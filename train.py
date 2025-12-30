"""
训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import time
from tqdm import tqdm
from typing import Dict
import matplotlib.pyplot as plt

from model import get_model
from dataset import create_dataloaders


class Trainer:
    """
    训练器类
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=7,  # 增加到7，避免过早降低学习率
            verbose=True,
            min_lr=1e-7
        )

        # 损失函数（MSE用于回归）
        self.criterion = nn.MSELoss()

        # 多任务学习权重（SOC和SOH的重要性）
        self.task_weights = {'soc': 1.0, 'soh': 1.0}  # 可以调整权重比例

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_soc_mae': [],
            'train_soh_mae': [],
            'val_soc_mae': [],
            'val_soh_mae': [],
            'learning_rate': []
        }

        # 最佳验证损失
        self.best_val_loss = float('inf')

        # 早停机制
        self.early_stop_patience = 5  # 5个epoch不改善就停止（快速方案）
        self.early_stop_counter = 0

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        epoch_loss = 0.0
        soc_mae_total = 0.0
        soh_mae_total = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='训练')
        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(X)

            # 计算损失 - 分别计算SOC和SOH损失
            soc_loss = self.criterion(outputs[:, 0], y[:, 0])
            soh_loss = self.criterion(outputs[:, 1], y[:, 1])
            loss = self.task_weights['soc'] * soc_loss + self.task_weights['soh'] * soh_loss

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            epoch_loss += loss.item()
            soc_mae = torch.abs(outputs[:, 0] - y[:, 0]).mean().item()
            soh_mae = torch.abs(outputs[:, 1] - y[:, 1]).mean().item()
            soc_mae_total += soc_mae
            soh_mae_total += soh_mae
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'soc_mae': f'{soc_mae:.4f}',
                'soh_mae': f'{soh_mae:.4f}'
            })

        return {
            'loss': epoch_loss / num_batches,
            'soc_mae': soc_mae_total / num_batches,
            'soh_mae': soh_mae_total / num_batches
        }

    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()

        epoch_loss = 0.0
        soc_mae_total = 0.0
        soh_mae_total = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='验证')
            for X, y in pbar:
                X, y = X.to(self.device), y.to(self.device)

                # 前向传播
                outputs = self.model(X)

                # 计算损失 - 分别计算SOC和SOH损失
                soc_loss = self.criterion(outputs[:, 0], y[:, 0])
                soh_loss = self.criterion(outputs[:, 1], y[:, 1])
                loss = self.task_weights['soc'] * soc_loss + self.task_weights['soh'] * soh_loss

                # 统计
                epoch_loss += loss.item()
                soc_mae = torch.abs(outputs[:, 0] - y[:, 0]).mean().item()
                soh_mae = torch.abs(outputs[:, 1] - y[:, 1]).mean().item()
                soc_mae_total += soc_mae
                soh_mae_total += soh_mae
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'soc_mae': f'{soc_mae:.4f}',
                    'soh_mae': f'{soh_mae:.4f}'
                })

        return {
            'loss': epoch_loss / num_batches,
            'soc_mae': soc_mae_total / num_batches,
            'soh_mae': soh_mae_total / num_batches
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'early_stop_counter': self.early_stop_counter
        }

        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到 {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.early_stop_counter = checkpoint.get('early_stop_counter', 0)  # 兼容旧检查点
        return checkpoint['epoch']

    def plot_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='训练损失')
        axes[0, 0].plot(self.history['val_loss'], label='验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # SOC MAE
        axes[0, 1].plot(self.history['train_soc_mae'], label='训练SOC MAE')
        axes[0, 1].plot(self.history['val_soc_mae'], label='验证SOC MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('SOC平均绝对误差')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # SOH MAE
        axes[1, 0].plot(self.history['train_soh_mae'], label='训练SOH MAE')
        axes[1, 0].plot(self.history['val_soh_mae'], label='验证SOH MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('SOH平均绝对误差')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 学习率
        axes[1, 1].plot(self.history['learning_rate'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('学习率变化')
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300)
        print(f"训练历史图表保存到 {os.path.join(self.save_dir, 'training_history.png')}")

    def train(self, num_epochs: int, start_epoch: int = 0):
        """训练模型"""
        if start_epoch > 0:
            print(f"从epoch {start_epoch}恢复训练，目标epoch {num_epochs}")
        else:
            print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        # 显示GPU信息
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        total_start_time = time.time()

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            # 训练
            train_metrics = self.train_epoch()
            print(f"训练 - Loss: {train_metrics['loss']:.4f}, "
                  f"SOC MAE: {train_metrics['soc_mae']:.4f}, "
                  f"SOH MAE: {train_metrics['soh_mae']:.4f}")

            # 验证
            val_metrics = self.validate()
            print(f"验证 - Loss: {val_metrics['loss']:.4f}, "
                  f"SOC MAE: {val_metrics['soc_mae']:.4f}, "
                  f"SOH MAE: {val_metrics['soh_mae']:.4f}")

            # 显示epoch耗时和预计剩余时间
            epoch_time = time.time() - epoch_start_time
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_time = epoch_time * remaining_epochs
            print(f"Epoch耗时: {epoch_time/60:.1f}分钟 | "
                  f"预计剩余: {estimated_time/60:.1f}分钟 ({estimated_time/3600:.1f}小时)")

            # 显示GPU内存使用
            if self.device == 'cuda':
                print(f"GPU显存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB / "
                      f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

            # 更新历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_soc_mae'].append(train_metrics['soc_mae'])
            self.history['train_soh_mae'].append(train_metrics['soh_mae'])
            self.history['val_soc_mae'].append(val_metrics['soc_mae'])
            self.history['val_soh_mae'].append(val_metrics['soh_mae'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # 学习率调度
            self.scheduler.step(val_metrics['loss'])

            # 保存检查点
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.early_stop_counter = 0  # 重置早停计数器
                print(f"验证损失改善! 新最佳: {self.best_val_loss:.6f}")
            else:
                self.early_stop_counter += 1
                print(f"验证损失未改善 ({self.early_stop_counter}/{self.early_stop_patience})")

            self.save_checkpoint(epoch + 1, is_best=is_best)

            # 早停检查
            if self.early_stop_counter >= self.early_stop_patience:
                print(f"\n早停触发！连续{self.early_stop_patience}个epoch验证损失未改善")
                print(f"最佳验证损失: {self.best_val_loss:.6f}")
                print(f"在epoch {epoch + 1}停止训练")
                break

        # 绘制训练历史
        self.plot_history()

        # 保存训练历史到JSON
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)

        total_time = time.time() - total_start_time
        print("\n" + "=" * 60)
        print(f"训练完成！总耗时: {total_time/60:.1f}分钟 ({total_time/3600:.2f}小时)")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print("=" * 60)


def main():
    """主函数"""
    # 配置
    config = {
        'batch_size': 384,  # 保守快速方案，确保不OOM
        'sequence_length': 60,
        'num_epochs': 15,  # 减少epoch数量加快训练
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 0,  # Windows下必须为0
        'model_type': 'lightweight',  # 使用轻量级模型加快训练
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'pin_memory': True  # 加速数据传输到GPU
    }

    print("配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, val_loader, test_loader, _, _ = create_dataloaders(
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        num_workers=config['num_workers']
    )

    # 创建模型
    print("\n创建模型...")
    model = get_model(
        model_type=config['model_type'],
        input_dim=6,
        d_model=64,  # 轻量级模型参数
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 尝试从断点恢复
    start_epoch = 0
    checkpoint_path = os.path.join('checkpoints', 'latest_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"\n发现检查点：{checkpoint_path}")
        try:
            start_epoch = trainer.load_checkpoint(checkpoint_path)
            print(f"从epoch {start_epoch}恢复训练")
            print(f"历史最佳验证损失: {trainer.best_val_loss:.6f}")
        except Exception as e:
            print(f"加载检查点失败: {e}")
            print("将从头开始训练")
            start_epoch = 0

    # 训练
    trainer.train(num_epochs=config['num_epochs'], start_epoch=start_epoch)


if __name__ == '__main__':
    main()

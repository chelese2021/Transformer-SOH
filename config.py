"""
配置文件
集中管理所有训练和模型参数
"""

# 数据配置
DATA_CONFIG = {
    'data_dir': 'data',
    'sequence_length': 60,  # 时间序列长度（60 * 10秒 = 10分钟）
    'prediction_horizon': 1,  # 预测未来多少步
    'train_ratio': 0.7,  # 训练集比例
    'val_ratio': 0.15,  # 验证集比例
    # test_ratio = 1 - train_ratio - val_ratio = 0.15
}

# 模型配置
MODEL_CONFIG = {
    'model_type': 'standard',  # 'standard' 或 'lightweight'
    'input_dim': 6,  # 输入特征维度
    'd_model': 128,  # Transformer模型维度
    'nhead': 8,  # 多头注意力头数
    'num_encoder_layers': 4,  # 编码器层数
    'dim_feedforward': 512,  # 前馈网络维度
    'dropout': 0.1,  # Dropout比例
}

# 轻量级模型配置
LIGHTWEIGHT_MODEL_CONFIG = {
    'model_type': 'lightweight',
    'input_dim': 6,
    'd_model': 64,
    'nhead': 4,
    'num_encoder_layers': 2,
    'dim_feedforward': 256,
    'dropout': 0.1,
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 128,  # 批次大小
    'num_epochs': 50,  # 训练轮数
    'learning_rate': 1e-4,  # 学习率
    'weight_decay': 1e-5,  # 权重衰减
    'num_workers': 4,  # 数据加载线程数
    'device': 'auto',  # 'auto', 'cuda', 'cpu'
    'save_dir': 'checkpoints',  # 模型保存目录
    'log_interval': 10,  # 日志打印间隔
}

# 优化器配置
OPTIMIZER_CONFIG = {
    'type': 'AdamW',  # 优化器类型
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
}

# 学习率调度器配置
SCHEDULER_CONFIG = {
    'type': 'ReduceLROnPlateau',  # 调度器类型
    'mode': 'min',
    'factor': 0.5,  # 学习率衰减因子
    'patience': 5,  # 容忍验证损失不下降的epoch数
    'verbose': True,
    'min_lr': 1e-7,
}

# 评估配置
EVAL_CONFIG = {
    'batch_size': 256,  # 评估批次大小（可以更大）
    'num_workers': 4,
    'save_dir': 'results',  # 结果保存目录
}

# 推理配置
INFERENCE_CONFIG = {
    'model_path': 'checkpoints/best_model.pth',
    'feature_scaler_path': 'feature_scaler.pkl',
    'target_scaler_path': 'target_scaler.pkl',
    'device': 'auto',
}

# 特征名称
FEATURE_NAMES = [
    'Charging_Current',
    'Max_Cell_Voltage',
    'Min_Cell_Voltage',
    'Max_Cell_Temperature',
    'Min_Cell_Temperature',
    'mileage'
]

# 目标名称
TARGET_NAMES = ['SOC', 'soh']


def get_config(config_type='train'):
    """
    获取配置

    参数：
        config_type: 配置类型 ('train', 'eval', 'inference')

    返回：
        配置字典
    """
    configs = {
        'train': {
            'data': DATA_CONFIG,
            'model': MODEL_CONFIG,
            'train': TRAIN_CONFIG,
            'optimizer': OPTIMIZER_CONFIG,
            'scheduler': SCHEDULER_CONFIG,
        },
        'eval': {
            'data': DATA_CONFIG,
            'model': MODEL_CONFIG,
            'eval': EVAL_CONFIG,
        },
        'inference': {
            'inference': INFERENCE_CONFIG,
        }
    }

    return configs.get(config_type, configs['train'])


def print_config(config):
    """打印配置"""
    print("=" * 70)
    print("配置信息")
    print("=" * 70)
    for category, params in config.items():
        print(f"\n{category.upper()}:")
        print("-" * 70)
        for key, value in params.items():
            print(f"  {key:25s}: {value}")
    print("=" * 70)


if __name__ == '__main__':
    # 打印训练配置
    config = get_config('train')
    print_config(config)

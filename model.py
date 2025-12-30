"""
Transformer模型用于电池SOC和SOH预测
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    为时间序列添加位置信息
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class BatteryTransformer(nn.Module):
    """
    基于Transformer的电池SOC和SOH预测模型

    架构：
        1. 输入嵌入层：将原始特征映射到d_model维度
        2. 位置编码：添加时间序列位置信息
        3. Transformer编码器：提取时间序列特征
        4. 双输出头：分别预测SOC和SOH

    参数：
        input_dim: 输入特征维度
        d_model: Transformer模型维度
        nhead: 多头注意力头数
        num_encoder_layers: 编码器层数
        dim_feedforward: 前馈网络维度
        dropout: Dropout比例
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # [seq_len, batch, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 全局池化（对时间维度）
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # SOC预测头
        self.soc_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # SOH预测头
        self.soh_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
            x: [batch_size, seq_len, input_dim]

        返回：
            output: [batch_size, 2] (SOC, SOH)
        """
        # x: [batch_size, seq_len, input_dim] -> [seq_len, batch_size, input_dim]
        x = x.transpose(0, 1)

        # 输入嵌入
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        # x: [seq_len, batch_size, d_model]

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)
        # x: [seq_len, batch_size, d_model]

        # 转置并池化
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch_size, d_model]

        # 预测SOC和SOH
        soc = self.soc_head(x)  # [batch_size, 1]
        soh = self.soh_head(x)  # [batch_size, 1]

        # 合并输出
        output = torch.cat([soc, soh], dim=1)  # [batch_size, 2]

        return output


class LightweightBatteryTransformer(nn.Module):
    """
    轻量级Transformer模型
    适用于资源受限的场景
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim

        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # SOC和SOH
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
            x: [batch_size, seq_len, input_dim]

        返回：
            output: [batch_size, 2]
        """
        x = x.transpose(0, 1)  # [seq_len, batch_size, input_dim]

        # 嵌入和位置编码
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 使用最后一个时间步的输出
        x = x[-1, :, :]  # [batch_size, d_model]

        # 预测
        output = self.output_layer(x)  # [batch_size, 2]

        return output


def get_model(model_type: str = 'standard', **kwargs) -> nn.Module:
    """
    获取模型实例

    参数：
        model_type: 'standard' 或 'lightweight'
        **kwargs: 模型参数

    返回：
        模型实例
    """
    if model_type == 'standard':
        return BatteryTransformer(**kwargs)
    elif model_type == 'lightweight':
        return LightweightBatteryTransformer(**kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


if __name__ == '__main__':
    # 测试模型
    batch_size = 32
    seq_len = 60
    input_dim = 6

    # 创建随机输入
    x = torch.randn(batch_size, seq_len, input_dim)

    # 标准模型
    print("标准Transformer模型:")
    model = BatteryTransformer(input_dim=input_dim)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n轻量级Transformer模型:")
    model_light = LightweightBatteryTransformer(input_dim=input_dim)
    output_light = model_light(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output_light.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model_light.parameters()):,}")

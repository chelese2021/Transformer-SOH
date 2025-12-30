"""
实时推理脚本
用于部署后的实时电池状态预测
"""
import torch
import numpy as np
import pickle
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from model import get_model


class BatteryPredictor:
    """
    电池状态预测器
    用于实时预测SOC和SOH
    """

    def __init__(
        self,
        model_path: str = 'checkpoints/best_model.pth',
        feature_scaler_path: str = 'feature_scaler.pkl',
        target_scaler_path: str = 'target_scaler.pkl',
        device: str = None
    ):
        """
        初始化预测器

        参数：
            model_path: 模型权重路径
            feature_scaler_path: 特征标准化器路径
            target_scaler_path: 目标标准化器路径
            device: 计算设备
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"加载模型到 {self.device}...")

        # 加载模型
        self.model = get_model(
            model_type='standard',
            input_dim=6,
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.1
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 加载标准化器
        with open(feature_scaler_path, 'rb') as f:
            self.feature_scaler = pickle.load(f)

        with open(target_scaler_path, 'rb') as f:
            self.target_scaler = pickle.load(f)

        print("模型加载完成！")

    def predict(
        self,
        charging_current: List[float],
        max_cell_voltage: List[float],
        min_cell_voltage: List[float],
        max_cell_temperature: List[float],
        min_cell_temperature: List[float],
        mileage: List[float]
    ) -> Dict[str, float]:
        """
        预测电池SOC和SOH

        参数：
            charging_current: 充电电流序列（长度60）
            max_cell_voltage: 最大单体电压序列（长度60）
            min_cell_voltage: 最小单体电压序列（长度60）
            max_cell_temperature: 最大单体温度序列（长度60）
            min_cell_temperature: 最小单体温度序列（长度60）
            mileage: 里程序列（长度60）

        返回：
            {'SOC': float, 'SOH': float}
        """
        # 验证输入长度
        seq_len = 60
        assert len(charging_current) == seq_len, f"充电电流序列长度应为{seq_len}"
        assert len(max_cell_voltage) == seq_len, f"最大单体电压序列长度应为{seq_len}"
        assert len(min_cell_voltage) == seq_len, f"最小单体电压序列长度应为{seq_len}"
        assert len(max_cell_temperature) == seq_len, f"最大单体温度序列长度应为{seq_len}"
        assert len(min_cell_temperature) == seq_len, f"最小单体温度序列长度应为{seq_len}"
        assert len(mileage) == seq_len, f"里程序列长度应为{seq_len}"

        # 构建输入特征
        features = np.column_stack([
            charging_current,
            max_cell_voltage,
            min_cell_voltage,
            max_cell_temperature,
            min_cell_temperature,
            mileage
        ])  # [seq_len, 6]

        # 标准化
        features = self.feature_scaler.transform(features)

        # 转换为张量
        X = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # [1, seq_len, 6]

        # 预测
        with torch.no_grad():
            output = self.model(X)  # [1, 2]
            output = output.cpu().numpy()

        # 反标准化
        output = self.target_scaler.inverse_transform(output)[0]

        return {
            'SOC': float(output[0]),
            'SOH': float(output[1])
        }

    def predict_from_array(self, data: np.ndarray) -> Dict[str, float]:
        """
        从numpy数组预测

        参数：
            data: [seq_len, 6] 的numpy数组
                  列顺序: [充电电流, 最大电压, 最小电压, 最大温度, 最小温度, 里程]

        返回：
            {'SOC': float, 'SOH': float}
        """
        assert data.shape == (60, 6), f"输入数据形状应为(60, 6)，实际为{data.shape}"

        # 标准化
        data = self.feature_scaler.transform(data)

        # 转换为张量
        X = torch.FloatTensor(data).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(X)
            output = output.cpu().numpy()

        # 反标准化
        output = self.target_scaler.inverse_transform(output)[0]

        return {
            'SOC': float(output[0]),
            'SOH': float(output[1])
        }


def demo():
    """演示如何使用预测器"""
    print("=" * 70)
    print("电池状态预测演示")
    print("=" * 70)

    # 创建预测器
    predictor = BatteryPredictor()

    # 模拟输入数据（60个时间步，每步10秒）
    print("\n生成模拟数据...")
    seq_len = 60

    # 模拟一个充电过程
    charging_current = [-17.5] * seq_len
    max_cell_voltage = [4.05 + i * 0.001 for i in range(seq_len)]
    min_cell_voltage = [4.03 + i * 0.001 for i in range(seq_len)]
    max_cell_temperature = [25.0 + np.random.randn() * 0.5 for _ in range(seq_len)]
    min_cell_temperature = [23.0 + np.random.randn() * 0.5 for _ in range(seq_len)]
    mileage = [120000.0] * seq_len

    # 预测
    print("\n进行预测...")
    result = predictor.predict(
        charging_current=charging_current,
        max_cell_voltage=max_cell_voltage,
        min_cell_voltage=min_cell_voltage,
        max_cell_temperature=max_cell_temperature,
        min_cell_temperature=min_cell_temperature,
        mileage=mileage
    )

    print("\n预测结果:")
    print("-" * 70)
    print(f"  SOC (充电状态): {result['SOC']:.2f}%")
    print(f"  SOH (健康状态): {result['SOH']:.4f} ({result['SOH']*100:.2f}%)")
    print("-" * 70)

    # 解释结果
    print("\n结果解释:")
    if result['SOC'] < 20:
        print("  ⚠️  电池电量较低，建议充电")
    elif result['SOC'] < 80:
        print("  ✓  电池电量正常")
    else:
        print("  ✓  电池电量充足")

    if result['SOH'] < 0.8:
        print("  ⚠️  电池健康状态较差，建议检查或更换")
    elif result['SOH'] < 0.9:
        print("  ⚠️  电池健康状态一般，需要关注")
    else:
        print("  ✓  电池健康状态良好")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    demo()

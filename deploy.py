"""
模型部署和推理模块
支持PyTorch模型和ONNX模型的实时推理
"""
import torch
import numpy as np
import pickle
import os
from typing import Union, Tuple
import time

from model import MultiScaleSOHSOCPredictor


class BatterySOHPredictor:
    """
    电池SOH/SOC预测器（部署版本）

    支持：
    1. PyTorch模型推理
    2. ONNX模型推理（边缘端部署）
    3. 批量和单样本推理
    """

    def __init__(
        self,
        model_path: str = 'checkpoints/best_model.pth',
        feature_scaler_path: str = 'feature_scaler.pkl',
        target_scaler_path: str = 'target_scaler.pkl',
        device: str = 'cuda',
        use_onnx: bool = False
    ):
        """
        初始化预测器

        参数：
            model_path: 模型权重路径或ONNX模型路径
            feature_scaler_path: 特征标准化器路径
            target_scaler_path: 目标标准化器路径
            device: 'cuda' 或 'cpu'
            use_onnx: 是否使用ONNX模型
        """
        self.device = device
        self.use_onnx = use_onnx

        # 加载标准化器
        print("加载标准化器...")
        with open(feature_scaler_path, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(target_scaler_path, 'rb') as f:
            self.target_scaler = pickle.load(f)
        print("✓ 标准化器加载完成")

        # 加载模型
        if use_onnx:
            self._load_onnx_model(model_path)
        else:
            self._load_pytorch_model(model_path)

    def _load_pytorch_model(self, model_path: str):
        """加载PyTorch模型"""
        print(f"加载PyTorch模型: {model_path}")

        # 创建模型实例
        self.model = MultiScaleSOHSOCPredictor(input_dim=6, d_model=64)

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print(f"✓ PyTorch模型加载完成")
        print(f"  设备: {self.device}")
        print(f"  参数量: {self.model.count_parameters():,}")

    def _load_onnx_model(self, model_path: str):
        """加载ONNX模型"""
        try:
            import onnxruntime as ort
            print(f"加载ONNX模型: {model_path}")

            # 创建推理会话
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(model_path, providers=providers)

            print(f"✓ ONNX模型加载完成")
            print(f"  推理设备: {self.ort_session.get_providers()}")

        except ImportError:
            raise ImportError("使用ONNX模型需要安装onnxruntime: pip install onnxruntime 或 pip install onnxruntime-gpu")

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        """
        预处理输入数据

        参数：
            data: 原始数据 [seq_len, features] 或 [batch, seq_len, features]

        返回：
            标准化后的tensor
        """
        # 处理单样本情况
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # [1, seq_len, features]

        # 标准化每个样本
        batch_size, seq_len, n_features = data.shape

        # 重塑为2D进行标准化
        data_2d = data.reshape(-1, n_features)
        data_normalized = self.feature_scaler.transform(data_2d)

        # 重塑回3D
        data_normalized = data_normalized.reshape(batch_size, seq_len, n_features)

        # 转换为tensor
        if not self.use_onnx:
            return torch.FloatTensor(data_normalized).to(self.device)
        else:
            return data_normalized.astype(np.float32)

    def postprocess(self, predictions: np.ndarray) -> dict:
        """
        后处理预测结果

        参数：
            predictions: 模型输出 [batch, 2]

        返回：
            dict: {'soc': [...], 'soh': [...]}
        """
        # 反标准化
        predictions = self.target_scaler.inverse_transform(predictions)

        # 确保在合理范围内
        soc = np.clip(predictions[:, 0], 0, 100)
        soh = np.clip(predictions[:, 1], 0, 100)

        return {
            'soc': soc.tolist(),
            'soh': soh.tolist()
        }

    def predict(self, data: np.ndarray, return_scale_weights: bool = False) -> dict:
        """
        预测SOC和SOH

        参数：
            data: 输入数据 [seq_len, features] 或 [batch, seq_len, features]
                 features顺序: [Charging_Current, Max_Cell_Voltage, Min_Cell_Voltage,
                               Max_Cell_Temperature, Min_Cell_Temperature, mileage]
            return_scale_weights: 是否返回多尺度权重（仅PyTorch模型）

        返回：
            结果字典，包含 'soc' 和 'soh'
        """
        # 预处理
        input_data = self.preprocess(data)

        # 推理
        if self.use_onnx:
            predictions = self._predict_onnx(input_data)
        else:
            predictions = self._predict_pytorch(input_data)

        # 后处理
        results = self.postprocess(predictions)

        # 添加多尺度权重（如果需要）
        if return_scale_weights and not self.use_onnx:
            if hasattr(self.model, 'last_scale_weights'):
                scale_weights = self.model.last_scale_weights.cpu().numpy()
                results['scale_weights'] = {
                    'short_term': scale_weights[:, 0].tolist(),
                    'mid_term': scale_weights[:, 1].tolist(),
                    'long_term': scale_weights[:, 2].tolist()
                }

        return results

    def _predict_pytorch(self, input_tensor: torch.Tensor) -> np.ndarray:
        """PyTorch模型推理"""
        with torch.no_grad():
            outputs = self.model(input_tensor)
        return outputs.cpu().numpy()

    def _predict_onnx(self, input_array: np.ndarray) -> np.ndarray:
        """ONNX模型推理"""
        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: input_array})
        return outputs[0]

    def predict_single(
        self,
        charging_current: float,
        max_cell_voltage: float,
        min_cell_voltage: float,
        max_cell_temp: float,
        min_cell_temp: float,
        mileage: float,
        sequence_length: int = 60
    ) -> dict:
        """
        单时间点预测（需要历史序列）

        参数：
            所有参数为当前时刻和历史时刻的值
            sequence_length: 序列长度

        注意：
            此函数假设输入是重复的单点数据
            实际使用时应传入真实的历史序列
        """
        # 构造输入序列（这里简化为重复当前值）
        # 实际应用中应该使用真实的历史数据
        features = np.array([
            charging_current,
            max_cell_voltage,
            min_cell_voltage,
            max_cell_temp,
            min_cell_temp,
            mileage
        ])

        # 重复以形成序列
        sequence = np.tile(features, (sequence_length, 1))  # [seq_len, 6]

        # 预测
        results = self.predict(sequence)

        # 返回单个值
        return {
            'soc': results['soc'][0],
            'soh': results['soh'][0]
        }

    def benchmark(self, num_iterations: int = 100, batch_size: int = 32, seq_len: int = 60):
        """
        性能基准测试

        参数：
            num_iterations: 迭代次数
            batch_size: 批次大小
            seq_len: 序列长度
        """
        print(f"\n性能基准测试")
        print("="*60)
        print(f"迭代次数: {num_iterations}")
        print(f"批次大小: {batch_size}")
        print(f"序列长度: {seq_len}")
        print(f"推理引擎: {'ONNX' if self.use_onnx else 'PyTorch'}")
        print(f"设备: {self.device}")

        # 生成随机数据
        dummy_data = np.random.randn(batch_size, seq_len, 6).astype(np.float32)

        # 预热
        print("\n预热中...")
        for _ in range(10):
            self.predict(dummy_data)

        # 基准测试
        print("测试中...")
        inference_times = []

        for _ in range(num_iterations):
            start_time = time.time()
            self.predict(dummy_data)
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)

        # 统计
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        throughput = (batch_size * 1000) / avg_time  # 样本/秒

        print("\n结果:")
        print("="*60)
        print(f"平均推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"最小推理时间: {min_time:.2f} ms")
        print(f"最大推理时间: {max_time:.2f} ms")
        print(f"吞吐量: {throughput:.1f} 样本/秒")
        print(f"单样本推理时间: {avg_time/batch_size:.3f} ms")
        print("="*60)


def deploy_to_edge():
    """
    边缘端部署流程示例
    """
    print("\n边缘端部署流程")
    print("="*80)

    from model import MultiScaleSOHSOCPredictor
    from model_compression import export_to_onnx, quantize_model_dynamic, prune_model

    # 1. 加载训练好的模型
    print("\n步骤1: 加载训练好的模型")
    model = MultiScaleSOHSOCPredictor(input_dim=6, d_model=64)
    if os.path.exists('checkpoints/best_model.pth'):
        checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型加载完成，参数量: {model.count_parameters():,}")
    else:
        print("⚠ 未找到训练好的模型")
        return

    # 2. 模型压缩（可选）
    print("\n步骤2: 模型压缩（可选）")
    print("2a. 剪枝...")
    pruned_model = prune_model(model, amount=0.2)  # 剪枝20%

    print("\n2b. 量化...")
    quantized_model = quantize_model_dynamic(pruned_model)

    # 3. 导出ONNX
    print("\n步骤3: 导出ONNX格式")
    dummy_input = torch.randn(1, 60, 6)
    export_to_onnx(
        quantized_model,
        dummy_input,
        save_path='deployment/battery_model_optimized.onnx',
        input_names=['battery_features'],
        output_names=['soc_soh_predictions']
    )

    # 4. 测试部署模型
    print("\n步骤4: 测试部署模型")
    predictor = BatterySOHPredictor(
        model_path='deployment/battery_model_optimized.onnx',
        use_onnx=True,
        device='cpu'
    )

    # 性能测试
    predictor.benchmark(num_iterations=100, batch_size=1, seq_len=60)

    print("\n✓ 边缘端部署完成！")
    print("模型文件: deployment/battery_model_optimized.onnx")
    print("可部署到: 嵌入式设备、移动端、边缘计算设备")


if __name__ == '__main__':
    """示例用法"""

    print("电池SOH/SOC预测器 - 部署示例")
    print("="*80)

    # 示例1: PyTorch模型推理
    print("\n示例1: PyTorch模型推理")
    print("-"*80)

    if os.path.exists('checkpoints/best_model.pth'):
        predictor = BatterySOHPredictor(
            model_path='checkpoints/best_model.pth',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # 生成示例数据
        dummy_data = np.random.randn(60, 6)  # [seq_len, features]

        # 预测
        results = predictor.predict(dummy_data, return_scale_weights=True)
        print(f"SOC预测: {results['soc'][0]:.2f}%")
        print(f"SOH预测: {results['soh'][0]:.2f}%")

        if 'scale_weights' in results:
            print(f"多尺度权重:")
            print(f"  短期: {results['scale_weights']['short_term'][0]:.3f}")
            print(f"  中期: {results['scale_weights']['mid_term'][0]:.3f}")
            print(f"  长期: {results['scale_weights']['long_term'][0]:.3f}")

        # 性能测试
        predictor.benchmark(num_iterations=100, batch_size=32)
    else:
        print("⚠ 未找到训练好的模型，跳过PyTorch推理示例")

    # 示例2: 边缘端部署
    print("\n\n示例2: 边缘端部署")
    print("-"*80)

    # 检查是否需要运行部署流程
    response = input("是否运行边缘端部署流程？(y/n): ")
    if response.lower() == 'y':
        os.makedirs('deployment', exist_ok=True)
        deploy_to_edge()

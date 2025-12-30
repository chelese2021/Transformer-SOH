"""
模型压缩和优化工具
支持量化、剪枝、ONNX导出，用于边缘端部署
"""
import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.nn.utils.prune as prune
import os
from typing import Optional
import numpy as np


def quantize_model_dynamic(model: nn.Module) -> nn.Module:
    """
    动态量化（Dynamic Quantization）

    优点：
    - 推理时动态量化激活值
    - 权重静态量化为INT8
    - 无需校准数据
    - 模型大小减少约4倍

    适用场景：
    - 模型权重是主要内存瓶颈
    - LSTM、Transformer等循环模型

    参数：
        model: 待量化的模型

    返回：
        量化后的模型
    """
    model.eval()

    # 动态量化：只量化Linear层
    quantized_model = quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    print("动态量化完成")
    print(f"量化层类型: Linear -> INT8")

    return quantized_model


def quantize_model_static(
    model: nn.Module,
    calibration_data: torch.Tensor,
    backend: str = 'fbgemm'
) -> nn.Module:
    """
    静态量化（Static Quantization）

    优点：
    - 权重和激活值都量化为INT8
    - 推理速度最快
    - 模型大小最小

    缺点：
    - 需要校准数据
    - 精度损失可能略大

    参数：
        model: 待量化的模型
        calibration_data: 校准数据 [batch, seq_len, features]
        backend: 'fbgemm'(CPU) 或 'qnnpack'(移动端)

    返回：
        量化后的模型
    """
    model.eval()

    # 设置量化配置
    model.qconfig = quantization.get_default_qconfig(backend)

    # 准备量化
    model_prepared = quantization.prepare(model, inplace=False)

    # 使用校准数据进行校准
    print("正在使用校准数据...")
    with torch.no_grad():
        model_prepared(calibration_data)

    # 转换为量化模型
    model_quantized = quantization.convert(model_prepared, inplace=False)

    print("静态量化完成")
    print(f"后端: {backend}")

    return model_quantized


def prune_model(
    model: nn.Module,
    amount: float = 0.3,
    prune_type: str = 'l1_unstructured'
) -> nn.Module:
    """
    模型剪枝（Pruning）

    移除不重要的参数，减少模型大小和计算量

    参数：
        model: 待剪枝的模型
        amount: 剪枝比例（0-1），默认0.3表示移除30%参数
        prune_type: 剪枝类型
            - 'l1_unstructured': L1非结构化剪枝（默认）
            - 'random_unstructured': 随机非结构化剪枝
            - 'ln_structured': Ln结构化剪枝

    返回：
        剪枝后的模型
    """
    print(f"开始剪枝，移除 {amount*100:.1f}% 的参数")

    parameters_to_prune = []

    # 收集所有可剪枝的层
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))

    # 执行剪枝
    if prune_type == 'l1_unstructured':
        for module, param_name in parameters_to_prune:
            prune.l1_unstructured(module, name=param_name, amount=amount)
    elif prune_type == 'random_unstructured':
        for module, param_name in parameters_to_prune:
            prune.random_unstructured(module, name=param_name, amount=amount)
    else:
        raise ValueError(f"不支持的剪枝类型: {prune_type}")

    # 使剪枝永久化
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    print(f"剪枝完成，已移除约 {amount*100:.1f}% 的参数")

    return model


def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    save_path: str = "battery_model.onnx",
    input_names: list = None,
    output_names: list = None,
    dynamic_axes: dict = None,
    opset_version: int = 11
):
    """
    导出模型为ONNX格式

    ONNX是跨平台的模型格式，支持多种推理引擎：
    - ONNX Runtime (CPU/GPU)
    - TensorRT (NVIDIA GPU)
    - OpenVINO (Intel CPU/GPU)
    - Core ML (Apple设备)

    参数：
        model: 待导出的模型
        dummy_input: 示例输入 [batch, seq_len, features]
        save_path: 保存路径
        input_names: 输入名称列表
        output_names: 输出名称列表
        dynamic_axes: 动态轴配置
        opset_version: ONNX opset版本
    """
    model.eval()

    # 默认配置
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    print(f"导出ONNX模型到: {save_path}")
    print(f"输入形状: {dummy_input.shape}")

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,  # 常量折叠优化
        export_params=True
    )

    print(f"ONNX模型已保存到 {save_path}")

    # 显示文件大小
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"模型大小: {file_size:.2f} MB")


def compare_model_size(
    original_model: nn.Module,
    compressed_model: nn.Module,
    model_name: str = "模型"
):
    """
    对比原始模型和压缩模型的大小

    参数：
        original_model: 原始模型
        compressed_model: 压缩后的模型
        model_name: 模型名称
    """
    # 计算参数量
    original_params = sum(p.numel() for p in original_model.parameters())
    compressed_params = sum(p.numel() for p in compressed_model.parameters())

    # 保存并计算文件大小
    torch.save(original_model.state_dict(), 'temp_original.pth')
    torch.save(compressed_model.state_dict(), 'temp_compressed.pth')

    original_size = os.path.getsize('temp_original.pth') / 1024  # KB
    compressed_size = os.path.getsize('temp_compressed.pth') / 1024  # KB

    # 清理临时文件
    os.remove('temp_original.pth')
    os.remove('temp_compressed.pth')

    # 打印对比
    print(f"\n{model_name} 压缩对比:")
    print("=" * 60)
    print(f"参数量:")
    print(f"  原始: {original_params:,}")
    print(f"  压缩: {compressed_params:,}")
    print(f"  减少: {(1 - compressed_params/original_params)*100:.1f}%")
    print(f"\n文件大小:")
    print(f"  原始: {original_size:.2f} KB")
    print(f"  压缩: {compressed_size:.2f} KB")
    print(f"  减少: {(1 - compressed_size/original_size)*100:.1f}%")
    print("=" * 60)


def verify_onnx_model(onnx_path: str, dummy_input: torch.Tensor, torch_model: nn.Module):
    """
    验证ONNX模型的正确性

    参数：
        onnx_path: ONNX模型路径
        dummy_input: 测试输入
        torch_model: PyTorch原始模型
    """
    try:
        import onnx
        import onnxruntime as ort

        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型格式验证通过")

        # 创建推理会话
        ort_session = ort.InferenceSession(onnx_path)

        # PyTorch推理
        torch_model.eval()
        with torch.no_grad():
            torch_output = torch_model(dummy_input).numpy()

        # ONNX推理
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        # 对比结果
        max_diff = np.abs(torch_output - ort_output).max()
        mean_diff = np.abs(torch_output - ort_output).mean()

        print(f"✓ 输出对比:")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")

        if max_diff < 1e-4:
            print("✓ ONNX模型输出与PyTorch一致")
        else:
            print("⚠ ONNX模型输出与PyTorch有较大差异")

    except ImportError:
        print("警告: 未安装onnx或onnxruntime，跳过验证")
        print("安装命令: pip install onnx onnxruntime")


if __name__ == '__main__':
    """测试模型压缩功能"""
    from model import MultiScaleSOHSOCPredictor

    # 创建模型
    model = MultiScaleSOHSOCPredictor(input_dim=6, d_model=64)
    print(f"原始模型参数量: {model.count_parameters():,}")

    # 创建示例输入
    batch_size = 1
    seq_len = 60
    input_dim = 6
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    print("\n" + "="*60)
    print("1. 测试动态量化")
    print("="*60)
    quantized_model = quantize_model_dynamic(model)
    compare_model_size(model, quantized_model, "动态量化")

    print("\n" + "="*60)
    print("2. 测试剪枝")
    print("="*60)
    pruned_model = prune_model(model, amount=0.3)
    compare_model_size(model, pruned_model, "剪枝30%")

    print("\n" + "="*60)
    print("3. 测试ONNX导出")
    print("="*60)
    export_to_onnx(
        model,
        dummy_input,
        save_path="multi_scale_battery_model.onnx",
        input_names=['battery_features'],
        output_names=['soc_soh_predictions']
    )

    # 验证ONNX模型
    verify_onnx_model("multi_scale_battery_model.onnx", dummy_input, model)

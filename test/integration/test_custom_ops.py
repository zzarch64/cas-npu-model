#!/usr/bin/env python
"""
测试自定义量化算子示例

演示如何调用PyTorch中不存在的新算子
"""

import sys
import os

# 确保导入本地的cas_npu包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cas_npu
import torch
import numpy as np

def test_custom_quantize():
    """测试自定义量化算子"""
    print("=" * 60)
    print("测试自定义量化算子")
    print("=" * 60)
    
    device = torch.device("cas_npu:0")
    
    # 创建测试数据
    print("\n1. 创建测试数据")
    input_tensor = torch.randn(4, 4, device=device, dtype=torch.float32)
    print(f"输入tensor shape: {input_tensor.shape}")
    print(f"输入tensor dtype: {input_tensor.dtype}")
    print(f"输入tensor device: {input_tensor.device}")
    print(f"输入数据:\n{input_tensor.cpu()}")
    
    # 设置量化参数
    scale = 0.1  # 量化缩放因子
    zero_point = 0  # 量化零点
    
    print(f"\n量化参数:")
    print(f"  scale: {scale}")
    print(f"  zero_point: {zero_point}")
    
    # 调用自定义量化算子
    print("\n2. 调用自定义量化算子 torch.ops.cas_npu.custom_quantize")
    try:
        output = torch.ops.cas_npu.custom_quantize(input_tensor, scale, zero_point)
        
        print(f"输出tensor shape: {output.shape}")
        print(f"输出tensor dtype: {output.dtype}")
        print(f"输出tensor device: {output.device}")
        print(f"量化后的数据:\n{output.cpu()}")
        
        # 验证量化结果
        print("\n3. 验证量化结果")
        input_cpu = input_tensor.cpu().numpy()
        output_cpu = output.cpu().numpy()
        
        # 手动计算期望的量化值
        expected_quantized = np.round(input_cpu / scale + zero_point)
        expected_quantized = np.clip(expected_quantized, -128, 127).astype(np.int8)
        
        print(f"期望的量化值:\n{expected_quantized}")
        print(f"实际量化值:\n{output_cpu}")
        
        # 检查是否匹配
        if np.array_equal(output_cpu, expected_quantized):
            print("✓ 量化结果正确！")
        else:
            print("✗ 量化结果不匹配")
            print(f"差异:\n{output_cpu - expected_quantized}")
        
        # 测试反量化（验证量化-反量化过程）
        print("\n4. 测试反量化（验证量化-反量化过程）")
        dequantized = (output_cpu.astype(np.float32) - zero_point) * scale
        print(f"反量化后的数据:\n{dequantized}")
        print(f"原始数据:\n{input_cpu}")
        
        # 计算量化误差
        error = np.abs(input_cpu - dequantized)
        max_error = np.max(error)
        mean_error = np.mean(error)
        print(f"\n量化误差统计:")
        print(f"  最大误差: {max_error:.6f}")
        print(f"  平均误差: {mean_error:.6f}")
        
        print("\n✓ 自定义量化算子测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_quantize_different_params():
    """测试不同量化参数"""
    print("\n" + "=" * 60)
    print("测试不同量化参数")
    print("=" * 60)
    
    device = torch.device("cas_npu:0")
    
    # 测试不同的scale和zero_point
    test_cases = [
        {"scale": 0.01, "zero_point": 0, "name": "小scale，零点=0"},
        {"scale": 0.1, "zero_point": 128, "name": "中等scale，零点=128"},
        {"scale": 1.0, "zero_point": -128, "name": "大scale，零点=-128"},
    ]
    
    for case in test_cases:
        print(f"\n测试: {case['name']}")
        input_tensor = torch.randn(2, 2, device=device, dtype=torch.float32)
        
        try:
            output = torch.ops.cas_npu.custom_quantize(
                input_tensor, 
                case["scale"], 
                case["zero_point"]
            )
            print(f"  输入范围: [{input_tensor.min().item():.3f}, {input_tensor.max().item():.3f}]")
            print(f"  输出范围: [{output.min().item()}, {output.max().item()}]")
            print(f"  ✓ 成功")
        except Exception as e:
            print(f"  ✗ 失败: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("自定义算子测试")
    print("=" * 60)
    print("\n说明:")
    print("1. 使用 TORCH_LIBRARY 定义新命名空间和新算子")
    print("2. 使用 TORCH_LIBRARY_IMPL 为特定设备实现算子")
    print("3. 在Python中通过 torch.ops.namespace.operator 调用")
    print("=" * 60)
    
    success = test_custom_quantize()
    test_custom_quantize_different_params()
    
    print("\n" + "=" * 60)
    if success:
        print("所有测试通过！✓")
    else:
        print("测试失败！✗")
    print("=" * 60)

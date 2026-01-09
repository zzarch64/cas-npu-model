#!/usr/bin/env python
"""
Linear 层测试

测试内容:
1. Linear 层前向传播
2. 手动矩阵乘法验证
3. 添加偏置验证
4. Linear 层反向传播
5. 梯度验证
"""

import sys
import os
import torch
import torch.nn as nn

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入测试框架
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_framework import (
    ensure_cas_npu, TestConfig, VerbosityLevel, check_tensor, verify_tensor_match,
    print_section, print_step, create_arg_parser, run_test, analyze_nan_distribution
)


def test_linear_forward(config: TestConfig) -> bool:
    """测试 Linear 层前向传播"""
    print_step("Linear Layer Forward Pass", config)
    
    device = torch.device(config.device)
    
    # 创建 Linear 层
    linear = nn.Linear(768, 3072).to(device)
    x = torch.randn(2, 10, 768, device=device)
    
    print_step("Checking weight and bias", config)
    weight_result = check_tensor(linear.weight, "Weight", config)
    bias_result = check_tensor(linear.bias, "Bias", config)
    
    if weight_result['has_nan'] or weight_result['has_inf'] or bias_result['has_nan'] or bias_result['has_inf']:
        return False
    
    print_step("Checking input", config)
    input_result = check_tensor(x, "Input", config)
    if input_result['has_nan'] or input_result['has_inf']:
        return False
    
    # 使用 Linear 层的前向传播
    print_step("Using Linear layer forward pass", config)
    try:
        y = linear(x)
        result = check_tensor(y, "Linear output", config)
        
        if result['has_nan'] or result['has_inf']:
            return False
        
        return True
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Linear forward pass failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_manual_mm_operation(config: TestConfig) -> bool:
    """测试手动矩阵乘法（检查 mm 操作）"""
    print_step("Manual mm Operation Test", config)
    
    device = torch.device(config.device)
    
    linear = nn.Linear(768, 3072).to(device)
    x = torch.randn(2, 10, 768, device=device)
    
    x_flat = x.view(-1, 768)  # (20, 768)
    weight_t = linear.weight.t()  # (768, 3072)
    
    print_step("Input tensors for mm", config)
    check_tensor(x_flat, "x_flat", config)
    check_tensor(weight_t, "weight_t", config)
    
    print_step("Executing mm(x_flat, weight_t)", config)
    try:
        mm_result = torch.mm(x_flat, weight_t)
        result = check_tensor(mm_result, "mm result", config)
        
        if result['has_nan'] or result['has_inf']:
            return False
        
        return True
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ mm operation failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_bias_addition(config: TestConfig) -> bool:
    """测试添加偏置"""
    print_step("Bias Addition Test", config)
    
    device = torch.device(config.device)
    
    linear = nn.Linear(768, 3072).to(device)
    x = torch.randn(2, 10, 768, device=device)
    
    x_flat = x.view(-1, 768)
    weight_t = linear.weight.t()
    mm_result = torch.mm(x_flat, weight_t)
    
    print_step("Adding bias", config)
    try:
        mm_plus_bias = mm_result + linear.bias.unsqueeze(0)
        result = check_tensor(mm_plus_bias, "mm_result + bias", config)
        
        if result['has_nan'] or result['has_inf']:
            return False
        
        return True
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Adding bias failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_linear_backward(config: TestConfig) -> bool:
    """测试 Linear 层反向传播"""
    print_step("Linear Layer Backward Pass", config)
    
    device = torch.device(config.device)
    
    linear = nn.Linear(768, 3072).to(device)
    x = torch.randn(2, 10, 768, device=device)
    
    # 前向传播
    y = linear(x)
    loss = y.sum()
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  Loss: {loss.item():.6f}")
    
    # 检查 grad_fn
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  y.grad_fn: {y.grad_fn}")
        if y.grad_fn:
            print(f"    grad_fn type: {type(y.grad_fn).__name__}")
    
    # 检查 requires_grad 状态
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Weight requires_grad: {linear.weight.requires_grad}")
        print(f"  Bias requires_grad: {linear.bias.requires_grad}")
        print(f"  Weight grad before backward: {linear.weight.grad}")
        print(f"  Bias grad before backward: {linear.bias.grad}")
    
    # 手动计算期望的梯度
    print_step("Computing expected gradients manually", config)
    grad_output = torch.ones_like(y)
    check_tensor(grad_output, "grad_output", config)
    
    x_flat_for_grad = x.view(-1, x.size(-1))  # (20, 768)
    grad_output_flat = grad_output.view(-1, grad_output.size(-1))  # (20, 3072)
    
    try:
        expected_weight_grad = torch.mm(x_flat_for_grad.t(), grad_output_flat)  # (768, 3072)
        result = check_tensor(expected_weight_grad, "expected weight grad", config)
        if result['has_nan'] or result['has_inf']:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("    ✗ Expected gradient computation produces NaN/Inf!")
            return False
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"    ✗ Expected gradient computation failed: {e}")
        return False
    
    # 执行反向传播
    print_step("Executing loss.backward()", config)
    try:
        loss.backward()
        
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ✓ backward completed")
        
        if linear.weight.grad is None:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("  ⚠ Weight gradient is None")
            return False
        
        print_step("Checking gradient tensor initialization", config)
        grad_cpu = linear.weight.grad.cpu()
        zero_count = (grad_cpu == 0).sum().item()
        nan_count = torch.isnan(grad_cpu).sum().item()
        inf_count = torch.isinf(grad_cpu).sum().item()
        
        if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
            print(f"    Zero values: {zero_count}/{grad_cpu.numel()} ({zero_count/grad_cpu.numel()*100:.2f}%)")
            print(f"    NaN values: {nan_count}/{grad_cpu.numel()} ({nan_count/grad_cpu.numel()*100:.2f}%)")
            print(f"    Inf values: {inf_count}/{grad_cpu.numel()} ({inf_count/grad_cpu.numel()*100:.2f}%)")
        
        # 检查 NaN 的分布模式
        if nan_count > 0:
            if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
                analyze_nan_distribution(linear.weight.grad, "weight.grad", config)
            return False
        
        result = check_tensor(linear.weight.grad, "Weight gradient", config)
        if result['has_nan'] or result['has_inf']:
            return False
        
        if linear.bias.grad is not None:
            result = check_tensor(linear.bias.grad, "Bias gradient", config)
            if result['has_nan'] or result['has_inf']:
                return False
        
        return True
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Backward pass failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def main():
    parser = create_arg_parser("Linear Layer Test")
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_cas_npu()
    
    print_section("Linear Layer Test", config)
    
    results = []
    
    # 运行所有测试
    results.append(("Linear Forward", run_test(test_linear_forward, config, "Linear Forward Test")))
    results.append(("Manual MM", run_test(test_manual_mm_operation, config, "Manual MM Test")))
    results.append(("Bias Addition", run_test(test_bias_addition, config, "Bias Addition Test")))
    results.append(("Linear Backward", run_test(test_linear_backward, config, "Linear Backward Test")))
    
    # 汇总结果
    print_section("Test Summary", config)
    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        if config.verbosity.value >= VerbosityLevel.QUIET.value:
            print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print()
        if all_passed:
            print("All tests passed! ✓")
        else:
            print("Some tests failed! ✗")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

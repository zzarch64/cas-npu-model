#!/usr/bin/env python
"""
梯度计算测试

测试内容:
1. 梯度tensor创建过程
2. 梯度流动过程（前向和反向传播）
3. 梯度数值验证
4. 手动梯度计算验证
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


def test_gradient_creation(config: TestConfig) -> bool:
    """测试梯度tensor的创建过程"""
    print_step("Gradient Tensor Creation", config)
    
    device = torch.device(config.device)
    
    # 创建Linear层
    linear = nn.Linear(768, 3072).to(device)
    x = torch.randn(2, 10, 768, device=device)
    
    print_step("Before forward pass", config)
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  weight.grad: {linear.weight.grad}")
        print(f"  bias.grad: {linear.bias.grad}")
    
    # 前向传播
    y = linear(x)
    loss = y.sum()
    
    print_step("After forward pass, before backward", config)
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  weight.grad: {linear.weight.grad}")
        print(f"  bias.grad: {linear.bias.grad}")
    
    # 手动创建梯度tensor，看看是否有问题
    print_step("Manually creating gradient tensors", config)
    try:
        manual_weight_grad = torch.zeros_like(linear.weight)
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  Manual weight grad shape: {manual_weight_grad.shape}")
            print(f"  Manual weight grad device: {manual_weight_grad.device}")
        
        result = check_tensor(manual_weight_grad, "manual_weight_grad", config)
        if result['has_nan'] or result['has_inf']:
            return False
        
        # 检查零值
        cpu_grad = manual_weight_grad.cpu()
        zero_count = (cpu_grad == 0).sum().item()
        if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
            print(f"  Zero count: {zero_count}/{cpu_grad.numel()} ({zero_count/cpu_grad.numel()*100:.2f}%)")
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Failed to create manual gradient tensor: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False
    
    # 执行反向传播
    print_step("Executing backward pass", config)
    try:
        loss.backward()
        
        print_step("After backward pass", config)
        if linear.weight.grad is None:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("  ✗ weight.grad is None!")
            return False
        
        result = check_tensor(linear.weight.grad, "weight.grad", config)
        
        if result['has_nan'] or result['has_inf']:
            if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
                analyze_nan_distribution(linear.weight.grad, "weight.grad", config)
            return False
        
        # 检查零值
        cpu_grad = linear.weight.grad.cpu()
        zero_count = (cpu_grad == 0).sum().item()
        if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
            print(f"  Zero count: {zero_count}/{cpu_grad.numel()} ({zero_count/cpu_grad.numel()*100:.2f}%)")
        
        return True
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Backward pass failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_gradient_flow(config: TestConfig) -> bool:
    """测试梯度流动过程"""
    print_step("Gradient Flow Test", config)
    
    device = torch.device(config.device)
    
    # 创建简单的Linear层
    linear = nn.Linear(10, 5).to(device)
    x = torch.randn(2, 10, device=device, requires_grad=True)
    
    print_step("Initial State", config)
    check_tensor(linear.weight, "weight", config)
    check_tensor(linear.bias, "bias", config)
    check_tensor(x, "input", config)
    
    # 前向传播
    print_step("Forward Pass", config)
    y = linear(x)
    check_tensor(y, "output", config)
    
    # 计算损失
    loss = y.sum()
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  loss: {loss.item():.6f}")
    
    # 手动计算期望的梯度
    print_step("Expected Gradients", config)
    grad_output = torch.ones_like(y)  # loss = y.sum(), so grad_output = ones
    check_tensor(grad_output, "grad_output", config)
    
    # weight.grad应该等于 x.t() @ grad_output
    # 注意：Linear层使用weight.t()，所以实际梯度需要转置
    expected_weight_grad = torch.mm(x.t(), grad_output)  # shape: [10, 5]
    check_tensor(expected_weight_grad, "expected weight.grad (before transpose)", config)
    
    # bias.grad应该等于 grad_output.sum(0)
    expected_bias_grad = grad_output.sum(0)
    check_tensor(expected_bias_grad, "expected bias.grad", config)
    
    # 执行反向传播
    print_step("Backward Pass", config)
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"    y.grad_fn: {y.grad_fn}")
        if y.grad_fn:
            print(f"      type: {type(y.grad_fn).__name__}")
            print(f"      next_functions: {y.grad_fn.next_functions}")
    
    try:
        loss.backward()
        
        print_step("After backward", config)
        check_tensor(linear.weight.grad, "weight.grad", config)
        check_tensor(linear.bias.grad, "bias.grad", config)
        check_tensor(x.grad, "input.grad", config)
        
        # 比较实际梯度和期望梯度
        if linear.weight.grad is not None:
            print_step("Gradient Comparison", config)
            actual_cpu = linear.weight.grad.cpu()
            # Linear层使用weight.t()，所以梯度需要转置
            expected_cpu = expected_weight_grad.t().cpu()
            
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print(f"  Actual weight.grad shape: {actual_cpu.shape}")
                print(f"  Expected weight.grad shape: {expected_cpu.shape}")
            
            # 检查shape是否匹配
            if actual_cpu.shape != expected_cpu.shape:
                if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                    print(f"  ⚠ Shape mismatch! Cannot compare directly.")
                return False
            
            # 检查NaN的位置
            if torch.isnan(actual_cpu).any():
                if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
                    nan_mask = torch.isnan(actual_cpu)
                    nan_indices = torch.nonzero(nan_mask, as_tuple=False)
                    max_show = min(10, len(nan_indices))
                    print(f"  NaN positions (first {max_show}):")
                    for i, idx in enumerate(nan_indices[:max_show]):
                        if len(idx) == 2:
                            print(f"    [{idx[0].item()}, {idx[1].item()}]")
                return False
            
            # 验证梯度匹配
            matched, diff_info = verify_tensor_match(actual_cpu, expected_cpu, "weight.grad", config.tolerance, config)
            return matched
        
        return False
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Backward pass failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_gradient_verification(config: TestConfig) -> bool:
    """测试梯度验证（手动计算对比）"""
    print_step("Gradient Verification (Manual Computation)", config)
    
    device = torch.device(config.device)
    
    # 创建一个简单的Linear层
    linear = nn.Linear(10, 5).to(device)
    x = torch.randn(2, 10, device=device, requires_grad=True)
    
    # 前向传播
    y = linear(x)
    loss = y.sum()
    
    # 清零梯度
    if linear.weight.grad is not None:
        linear.weight.grad.zero_()
    if linear.bias.grad is not None:
        linear.bias.grad.zero_()
    
    # 反向传播
    try:
        loss.backward()
        
        if linear.weight.grad is None:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("  ✗ weight.grad is None!")
            return False
        
        # 手动计算期望的梯度
        # weight.grad应该等于 input.t() @ grad_output
        # 对于Linear层: y = x @ W.t() + b
        # weight.grad = x.t() @ grad_output (where grad_output = ones)
        # 注意：Linear层使用weight.t()，所以实际梯度需要转置
        grad_output = torch.ones_like(y)
        expected_weight_grad = torch.mm(x.t(), grad_output)  # shape: [10, 5]
        # Linear层使用weight.t()，所以梯度需要转置
        expected_weight_grad = expected_weight_grad.t()  # shape: [5, 10]
        
        if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
            print(f"    Expected weight grad shape: {expected_weight_grad.shape}")
            print(f"    Actual weight grad shape: {linear.weight.grad.shape}")
        
        expected_cpu = expected_weight_grad.cpu()
        actual_cpu = linear.weight.grad.cpu()
        
        # 检查是否匹配（忽略NaN）
        if torch.isnan(actual_cpu).any():
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("    ✗ Actual gradient contains NaN, cannot compare")
            return False
        
        matched, diff_info = verify_tensor_match(actual_cpu, expected_cpu, "weight.grad", config.tolerance, config)
        return matched
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Verification failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_manual_gradient_computation(config: TestConfig) -> bool:
    """测试手动梯度计算（验证mm操作在梯度计算中的使用）"""
    print_step("Manual Gradient Computation", config)
    
    device = torch.device(config.device)
    
    linear = nn.Linear(10, 5).to(device)
    x = torch.randn(2, 10, device=device, requires_grad=True)
    
    y = linear(x)
    grad_output = torch.ones_like(y)
    
    # 手动计算 weight.grad = x.t() @ grad_output
    print_step("Computing weight.grad = x.t() @ grad_output", config)
    try:
        x_t = x.t()
        check_tensor(x_t, "x.t()", config)
        check_tensor(grad_output, "grad_output", config)
        
        manual_weight_grad = torch.mm(x_t, grad_output)
        result = check_tensor(manual_weight_grad, "manual weight.grad", config)
        
        if result['has_nan'] or result['has_inf']:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("    ✗ Manual computation produces NaN!")
                print("    This suggests mm operation has a problem in backward context")
            return False
        else:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("    ✓ Manual computation OK")
            return True
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"    ✗ Manual computation failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def main():
    parser = create_arg_parser("Gradient Computation Test")
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_cas_npu()
    
    print_section("Gradient Computation Test", config)
    
    results = []
    
    # 运行所有测试
    results.append(("Gradient Creation", run_test(test_gradient_creation, config, "Gradient Creation Test")))
    results.append(("Gradient Flow", run_test(test_gradient_flow, config, "Gradient Flow Test")))
    results.append(("Gradient Verification", run_test(test_gradient_verification, config, "Gradient Verification Test")))
    results.append(("Manual Computation", run_test(test_manual_gradient_computation, config, "Manual Gradient Computation Test")))
    
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

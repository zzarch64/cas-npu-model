#!/usr/bin/env python
"""
addmm 操作测试

测试内容:
1. addmm 前向传播
2. addmm 梯度计算
3. 梯度数值验证（与手动计算对比）
4. 逐步检查梯度计算过程
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


def test_addmm_forward(config: TestConfig) -> bool:
    """测试 addmm 前向传播"""
    print_step("addmm Forward Pass", config)
    
    device = torch.device(config.device)
    
    # 创建 tensor（模拟 Linear 层的前向传播）
    # Linear 层使用: output = input @ weight.t() + bias
    # 这等价于: output = addmm(bias, input, weight.t())
    batch_size = 20
    in_features = 768
    out_features = 3072
    
    input_t = torch.randn(batch_size, in_features, device=device, requires_grad=True)
    weight = torch.randn(out_features, in_features, device=device, requires_grad=True)
    bias = torch.randn(out_features, device=device, requires_grad=True)
    
    print_step("Input tensors", config)
    check_tensor(input_t, "input", config)
    check_tensor(weight, "weight", config)
    check_tensor(bias, "bias", config)
    
    # 使用 addmm
    print_step("Executing addmm", config)
    try:
        output = torch.addmm(bias, input_t, weight.t())
        
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ✓ addmm completed")
        
        result = check_tensor(output, "output", config)
        
        if result['has_nan'] or result['has_inf']:
            return False
        
        return True
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ addmm failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_addmm_gradient(config: TestConfig) -> bool:
    """测试 addmm 操作的梯度计算"""
    print_step("addmm Gradient Computation", config)
    
    device = torch.device(config.device)
    
    batch_size = 20
    in_features = 768
    out_features = 3072
    
    input_t = torch.randn(batch_size, in_features, device=device, requires_grad=True)
    weight = torch.randn(out_features, in_features, device=device, requires_grad=True)
    bias = torch.randn(out_features, device=device, requires_grad=True)
    
    # 前向传播
    output = torch.addmm(bias, input_t, weight.t())
    result = check_tensor(output, "output", config)
    
    if result['has_nan'] or result['has_inf']:
        return False
    
    # 计算损失并反向传播
    print_step("Backward pass", config)
    loss = output.sum()
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  loss: {loss.item():.6f}")
    
    try:
        loss.backward()
        
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ✓ backward completed")
        
        # 检查梯度
        print_step("Gradient check", config)
        all_ok = True
        
        if weight.grad is not None:
            result = check_tensor(weight.grad, "weight.grad", config)
            
            if result['has_nan'] or result['has_inf']:
                if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
                    analyze_nan_distribution(weight.grad, "weight.grad", config)
                all_ok = False
        else:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("  ✗ weight.grad is None!")
            all_ok = False
        
        if bias.grad is not None:
            result = check_tensor(bias.grad, "bias.grad", config)
            if result['has_nan'] or result['has_inf']:
                all_ok = False
        else:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("  ✗ bias.grad is None!")
            all_ok = False
        
        return all_ok
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ backward failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_addmm_gradient_verification(config: TestConfig) -> bool:
    """测试 addmm 梯度验证（与手动计算对比）"""
    print_step("addmm Gradient Verification", config)
    
    device = torch.device(config.device)
    
    batch_size = 20
    in_features = 768
    out_features = 3072
    
    # 创建 tensor
    input_t = torch.randn(batch_size, in_features, device=device, requires_grad=True)
    weight = torch.randn(out_features, in_features, device=device, requires_grad=True)
    bias = torch.randn(out_features, device=device, requires_grad=True)
    
    # 前向传播
    output = torch.addmm(bias, input_t, weight.t())
    
    # 创建 grad_output（模拟上一层传来的梯度）
    print_step("Create grad_output", config)
    grad_output = torch.ones_like(output)
    check_tensor(grad_output, "grad_output", config)
    
    # 手动计算梯度（用于对比）
    print_step("Manual gradient computation", config)
    input_cpu = input_t.detach().cpu()
    weight_cpu = weight.detach().cpu()
    grad_output_cpu = grad_output.cpu()
    
    # 对于 addmm(bias, input, weight.t())，梯度计算：
    # weight.grad = grad_output.t() @ input
    expected_weight_grad = grad_output_cpu.t() @ input_cpu
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  expected_weight_grad shape: {expected_weight_grad.shape}")
    
    result = check_tensor(expected_weight_grad, "expected_weight_grad", config)
    if result['has_nan'] or result['has_inf']:
        return False
    
    # 使用 autograd 计算梯度
    print_step("Autograd backward", config)
    loss = output.sum()
    
    # 清零梯度
    if weight.grad is not None:
        weight.grad.zero_()
    if bias.grad is not None:
        bias.grad.zero_()
    if input_t.grad is not None:
        input_t.grad.zero_()
    
    try:
        loss.backward()
        
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ✓ backward completed")
        
        # 检查实际梯度
        print_step("Check actual gradients", config)
        if weight.grad is None:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("  ✗ weight.grad is None!")
            return False
        
        actual_weight_grad_cpu = weight.grad.cpu()
        result = check_tensor(weight.grad, "actual_weight_grad", config)
        
        if result['has_nan'] or result['has_inf']:
            # 分析 NaN/Inf 的分布
            if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
                analyze_nan_distribution(weight.grad, "actual_weight_grad", config)
                
                # 比较有效值
                valid_mask = ~torch.isnan(actual_weight_grad_cpu) & ~torch.isinf(actual_weight_grad_cpu)
                if valid_mask.any():
                    valid_actual = actual_weight_grad_cpu[valid_mask]
                    if expected_weight_grad.shape == actual_weight_grad_cpu.shape:
                        valid_expected = expected_weight_grad[valid_mask]
                        diff = (valid_actual - valid_expected).abs()
                        max_diff = diff.max().item()
                        mean_diff = diff.mean().item()
                        print(f"    Valid values comparison:")
                        print(f"      Valid count: {valid_mask.sum().item()}/{actual_weight_grad_cpu.numel()}")
                        print(f"      Max difference: {max_diff:.6f}")
                        print(f"      Mean difference: {mean_diff:.6f}")
            return False
        
        # 比较梯度
        if expected_weight_grad.shape == actual_weight_grad_cpu.shape:
            matched, diff_info = verify_tensor_match(
                actual_weight_grad_cpu, expected_weight_grad,
                "weight.grad", config.tolerance, config
            )
            return matched
        else:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print(f"  Warning: Shape mismatch! Expected {expected_weight_grad.shape}, got {actual_weight_grad_cpu.shape}")
            return False
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Test failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_addmm_step_by_step(config: TestConfig) -> bool:
    """逐步测试 addmm 操作的梯度计算"""
    print_step("addmm Step-by-Step Test", config)
    
    device = torch.device(config.device)
    
    batch_size = 20
    in_features = 768
    out_features = 3072
    
    # 创建 tensor
    print_step("Create input tensors", config)
    input_t = torch.randn(batch_size, in_features, device=device, requires_grad=True)
    weight = torch.randn(out_features, in_features, device=device, requires_grad=True)
    bias = torch.randn(out_features, device=device, requires_grad=True)
    
    check_tensor(input_t, "input_t", config)
    check_tensor(weight, "weight", config)
    check_tensor(bias, "bias", config)
    
    # 前向传播
    print_step("Forward pass", config)
    output = torch.addmm(bias, input_t, weight.t())
    result = check_tensor(output, "output", config)
    
    if result['has_nan'] or result['has_inf']:
        return False
    
    # 创建 grad_output
    print_step("Create grad_output", config)
    grad_output = torch.ones_like(output)
    check_tensor(grad_output, "grad_output", config)
    
    # 手动计算期望的梯度
    print_step("Manual gradient computation", config)
    input_cpu = input_t.detach().cpu()
    grad_output_cpu = grad_output.cpu()
    
    expected_weight_grad = grad_output_cpu.t() @ input_cpu
    check_tensor(expected_weight_grad, "expected_weight_grad", config)
    
    result = check_tensor(expected_weight_grad, "expected_weight_grad", config)
    if result['has_nan'] or result['has_inf']:
        return False
    
    # 使用 autograd 计算梯度
    print_step("Autograd backward", config)
    loss = output.sum()
    
    # 清零梯度
    if weight.grad is not None:
        weight.grad.zero_()
    if bias.grad is not None:
        bias.grad.zero_()
    if input_t.grad is not None:
        input_t.grad.zero_()
    
    try:
        loss.backward()
        
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ✓ backward completed")
        
        # 检查实际梯度
        print_step("Check actual gradients", config)
        if weight.grad is None:
            return False
        
        check_tensor(weight.grad, "weight.grad", config)
        
        actual_weight_grad_cpu = weight.grad.cpu()
        result = check_tensor(weight.grad, "weight.grad", config)
        
        if result['has_nan'] or result['has_inf']:
            if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
                analyze_nan_distribution(weight.grad, "weight.grad", config)
            return False
        
        # 比较梯度
        if expected_weight_grad.shape == actual_weight_grad_cpu.shape:
            matched, diff_info = verify_tensor_match(
                actual_weight_grad_cpu, expected_weight_grad,
                "weight.grad", config.tolerance, config
            )
            return matched
        else:
            return False
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Test failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def main():
    parser = create_arg_parser("addmm Operation Test")
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_cas_npu()
    
    print_section("addmm Operation Test", config)
    
    results = []
    
    # 运行所有测试
    results.append(("addmm Forward", run_test(test_addmm_forward, config, "addmm Forward Test")))
    results.append(("addmm Gradient", run_test(test_addmm_gradient, config, "addmm Gradient Test")))
    results.append(("addmm Gradient Verification", run_test(test_addmm_gradient_verification, config, "addmm Gradient Verification Test")))
    results.append(("addmm Step-by-Step", run_test(test_addmm_step_by_step, config, "addmm Step-by-Step Test")))
    
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

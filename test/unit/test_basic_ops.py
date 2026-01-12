#!/usr/bin/env python
"""
基础操作测试 - add_ 和 copy_ 操作

测试内容:
1. add_.Tensor 操作（原地加法）
2. 梯度累积模拟（使用 add_）
3. 大 tensor 拷贝（CPU <-> Device）
4. 部分拷贝（包含 NaN 的情况）
"""

import sys
import os
import torch

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入测试框架
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_framework import (
    ensure_echo_npu, TestConfig, VerbosityLevel, check_tensor, verify_tensor_match,
    print_section, print_step, create_arg_parser, run_test
)


def test_add_inplace(config: TestConfig) -> bool:
    """测试 add_.Tensor 操作"""
    print_step("Testing add_.Tensor", config)
    
    device = torch.device(config.device)
    
    # 创建两个 tensor
    a = torch.randn(3072, 768, device=device)
    b = torch.randn(3072, 768, device=device)
    
    print_step("Initial State", config)
    check_tensor(a, "a", config)
    check_tensor(b, "b", config)
    
    # 保存原始值用于验证
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    expected = a_cpu + b_cpu
    
    # 测试 add_.Tensor
    print_step("Executing a.add_(b)", config)
    try:
        a.add_(b)
        
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ✓ add_.Tensor completed")
        
        # 检查结果
        a_cpu_after = a.cpu()
        result = check_tensor(a, "result", config)
        
        if result['has_nan'] or result['has_inf']:
            return False
        
        # 验证结果
        matched, diff_info = verify_tensor_match(a_cpu_after, expected, "result", config.tolerance, config)
        return matched
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ add_.Tensor failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_gradient_accumulation(config: TestConfig) -> bool:
    """模拟梯度累积过程"""
    print_step("Gradient Accumulation Simulation", config)
    
    device = torch.device(config.device)
    
    # 模拟梯度 tensor（初始化为 0）
    grad = torch.zeros(3072, 768, device=device)
    
    print_step("Initial gradient tensor", config)
    check_tensor(grad, "grad", config)
    
    # 检查初始状态
    grad_cpu = grad.cpu()
    zero_count = (grad_cpu == 0).sum().item()
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  Zero count: {zero_count}/{grad_cpu.numel()}")
    
    # 模拟梯度累积（添加新的梯度）
    new_grad = torch.randn(3072, 768, device=device)
    print_step("Adding new gradient", config)
    check_tensor(new_grad, "new_grad", config)
    
    new_grad_cpu = new_grad.cpu()
    expected = grad_cpu + new_grad_cpu
    
    # 使用 add_ 累积梯度
    print_step("Accumulating gradient with add_", config)
    try:
        grad.add_(new_grad)
        
        # 检查结果
        grad_cpu_after = grad.cpu()
        result = check_tensor(grad, "accumulated_grad", config)
        
        if result['has_nan'] or result['has_inf']:
            if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
                if result['has_nan']:
                    nan_rows = torch.isnan(grad_cpu_after).any(dim=1).sum().item()
                    print(f"    NaN rows: {nan_rows}/{grad_cpu_after.shape[0]}")
            return False
        
        # 验证结果
        matched, diff_info = verify_tensor_match(grad_cpu_after, expected, "accumulated_grad", config.tolerance, config)
        return matched
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Gradient accumulation failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_copy_large_tensor(config: TestConfig) -> bool:
    """测试大 tensor 的拷贝"""
    print_step("Large Tensor Copy Test", config)
    
    device = torch.device(config.device)
    
    # 创建大 tensor（模拟梯度 tensor 的大小）
    size = (3072, 768)
    cpu_tensor = torch.randn(*size)
    
    print_step("Creating CPU tensor", config)
    check_tensor(cpu_tensor, "cpu_tensor", config)
    
    # 拷贝到设备
    print_step("Copying to device", config)
    try:
        device_tensor = cpu_tensor.to(device)
        
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ✓ Copy completed")
            print(f"  Device tensor shape: {device_tensor.shape}")
            print(f"  Is contiguous: {device_tensor.is_contiguous()}")
        
        # 检查设备 tensor
        device_cpu = device_tensor.cpu()
        result = check_tensor(device_tensor, "device_tensor", config)
        
        if result['has_nan'] or result['has_inf']:
            return False
        
        # 验证数据是否正确
        matched, diff_info = verify_tensor_match(device_cpu, cpu_tensor, "copied_tensor", config.tolerance, config)
        if not matched:
            return False
        
        # 测试反向拷贝（Device -> CPU）
        print_step("Copying back to CPU", config)
        device_tensor2 = torch.randn(*size, device=device)
        cpu_tensor2 = device_tensor2.cpu()
        
        result2 = check_tensor(cpu_tensor2, "cpu_tensor2", config)
        return not (result2['has_nan'] or result2['has_inf'])
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Copy failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def test_copy_partial(config: TestConfig) -> bool:
    """测试部分拷贝（模拟梯度计算过程中的情况）"""
    print_step("Partial Copy Test (with NaN)", config)
    
    device = torch.device(config.device)
    
    # 创建设备 tensor（模拟梯度 tensor）
    device_tensor = torch.zeros(3072, 768, device=device)
    
    print_step("Initial device tensor", config)
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  Shape: {device_tensor.shape}")
        print(f"  Is contiguous: {device_tensor.is_contiguous()}")
    
    # 创建 CPU tensor（模拟梯度计算结果）
    cpu_tensor = torch.randn(3072, 768)
    
    # 只更新部分行（模拟梯度计算过程中某些行没有被计算）
    cpu_tensor[100:200, :] = float('nan')  # 模拟某些行包含 NaN
    
    print_step("CPU tensor with NaN", config)
    nan_count = torch.isnan(cpu_tensor).sum().item()
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  NaN count: {nan_count}/{cpu_tensor.numel()}")
    
    # 拷贝到设备
    print_step("Copying to device", config)
    try:
        device_tensor.copy_(cpu_tensor)
        
        # 检查设备 tensor
        device_cpu = device_tensor.cpu()
        nan_count_after = torch.isnan(device_cpu).sum().item()
        
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  NaN count after copy: {nan_count_after}/{device_cpu.numel()}")
        
        if nan_count_after == nan_count:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("  ✓ NaN correctly copied")
            return True
        else:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print(f"  ✗ NaN count mismatch: expected {nan_count}, got {nan_count_after}")
            return False
        
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Copy failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


def main():
    parser = create_arg_parser("Basic Operations Test")
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_echo_npu()
    
    print_section("Basic Operations Test", config)
    
    results = []
    
    # 运行所有测试
    results.append(("add_.Tensor", run_test(test_add_inplace, config, "add_.Tensor Test")))
    results.append(("Gradient Accumulation", run_test(test_gradient_accumulation, config, "Gradient Accumulation Test")))
    results.append(("Large Tensor Copy", run_test(test_copy_large_tensor, config, "Large Tensor Copy Test")))
    results.append(("Partial Copy", run_test(test_copy_partial, config, "Partial Copy Test")))
    
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

#!/usr/bin/env python
"""
AddressSanitizer 测试脚本

只测试 masked_fill_ 相关的操作，避免加载完整模型

使用方法:
    python test/tools/test_asan.py
    python test/tools/test_asan.py -vv
"""

import sys
import os
import argparse

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入测试框架
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_framework import (
    ensure_cas_npu, TestConfig, VerbosityLevel, check_tensor, verify_tensor_match,
    print_section, print_step, create_arg_parser, run_test
)

import torch


def test_simple_masked_fill(config: TestConfig) -> bool:
    """测试简单的 masked_fill_"""
    print_step("Simple masked_fill_", config)
    
    device = torch.device(config.device)
    
    x = torch.ones((2, 3), dtype=torch.float32, device=device)
    mask = torch.tensor([[True, False, True], [False, True, False]], device=device)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Before: {x.cpu()}")
    
    x.masked_fill_(mask, -1.0)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  After: {x.cpu()}")
    
    # 验证结果
    expected = torch.tensor([[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]], dtype=torch.float32)
    matched, _ = verify_tensor_match(x.cpu(), expected, "masked_fill_", 1e-7, config)
    
    return matched


def test_attention_mask_processing(config: TestConfig) -> bool:
    """模拟 attention_mask 处理"""
    print_step("Attention mask processing", config)
    
    device = torch.device(config.device)
    
    seq_len = 10
    batch_size = 1
    
    # 创建 attention_mask
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device=device)
    
    # 创建 causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    
    # 合并 masks
    combined_mask = torch.full((batch_size, 1, seq_len, seq_len), float('-inf'), device=device)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  attention_mask shape: {attention_mask.shape}")
        print(f"  causal_mask shape: {causal_mask.shape}")
        print(f"  combined_mask shape: {combined_mask.shape}, dtype: {combined_mask.dtype}")
    
    # 这一步会调用 masked_fill_
    combined_mask = combined_mask.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  After masked_fill_: shape={combined_mask.shape}")
        print(f"  Sample values: {combined_mask[0, 0, :3, :3].cpu()}")
    
    # 验证结果：下三角应该是0，上三角应该是-inf
    lower_tri = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    upper_tri = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    
    lower_values = combined_mask[0, 0][lower_tri]
    upper_values = combined_mask[0, 0][upper_tri]
    
    lower_ok = (lower_values == 0.0).all().item()
    upper_ok = torch.isinf(upper_values).all().item()
    
    return lower_ok and upper_ok


def test_multiple_masked_fill(config: TestConfig) -> bool:
    """测试多次调用 masked_fill_"""
    print_step("Multiple masked_fill_ calls", config)
    
    device = torch.device(config.device)
    
    x = torch.ones((5, 5), dtype=torch.float32, device=device)
    for i in range(10):
        mask = torch.rand(5, 5, device=device) > 0.5
        x.masked_fill_(mask, float(i))
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  After 10 masked_fill_ calls: {x.cpu()}")
    
    # 验证没有 NaN 或 Inf
    result = check_tensor(x, "x after multiple masked_fill_", config)
    return not (result['has_nan'] or result['has_inf'])


def test_different_tensor_sizes(config: TestConfig) -> bool:
    """测试不同大小的 tensor"""
    print_step("Different tensor sizes", config)
    
    device = torch.device(config.device)
    
    sizes = [(1, 1), (10, 10), (100, 100), (1000, 1000)]
    all_passed = True
    
    for size in sizes:
        x = torch.ones(size, dtype=torch.float32, device=device)
        mask = torch.rand(size, device=device) > 0.5
        x.masked_fill_(mask, -1.0)
        
        result = check_tensor(x, f"x size {size}", config)
        if result['has_nan'] or result['has_inf']:
            all_passed = False
        elif config.verbosity.value >= VerbosityLevel.VERBOSE.value:
            print(f"    Size {size}: OK")
    
    return all_passed


def main():
    parser = create_arg_parser("AddressSanitizer Test")
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_cas_npu()
    
    print_section("AddressSanitizer Test", config)
    
    results = []
    
    results.append(("Simple masked_fill_", run_test(test_simple_masked_fill, config, "Simple masked_fill_ Test")))
    results.append(("Attention mask processing", run_test(test_attention_mask_processing, config, "Attention Mask Processing Test")))
    results.append(("Multiple masked_fill_", run_test(test_multiple_masked_fill, config, "Multiple masked_fill_ Test")))
    results.append(("Different tensor sizes", run_test(test_different_tensor_sizes, config, "Different Tensor Sizes Test")))
    
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

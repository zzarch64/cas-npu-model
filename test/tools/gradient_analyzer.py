#!/usr/bin/env python
"""
梯度 NaN 分析工具

分析梯度 tensor 中 NaN 的分布模式，帮助诊断梯度计算问题。

使用方法:
    python test/tools/gradient_analyzer.py [options]
"""

import sys
import os
import torch
import torch.nn as nn
import argparse

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入测试框架
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_framework import (
    ensure_echo_npu, TestConfig, VerbosityLevel, check_tensor, analyze_nan_distribution,
    print_section, print_step, create_arg_parser
)


def analyze_gradient_nan_pattern(config: TestConfig):
    """分析梯度 NaN 的分布模式"""
    print_section("Gradient NaN Pattern Analysis", config)
    
    device = torch.device(config.device)
    
    # 创建 Linear 层
    linear = nn.Linear(768, 3072).to(device)
    x = torch.randn(2, 10, 768, device=device)
    
    # 前向传播
    y = linear(x)
    loss = y.sum()
    
    # 反向传播
    print_step("Running backward pass", config)
    loss.backward()
    
    if linear.weight.grad is None:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("✗ weight.grad is None!")
        return
    
    grad_cpu = linear.weight.grad.cpu()
    nan_mask = torch.isnan(grad_cpu)
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"\nGradient shape: {grad_cpu.shape}")
        print(f"NaN count: {nan_mask.sum().item()}/{grad_cpu.numel()}")
    
    # 分析 NaN 的分布
    print_step("NaN Distribution Analysis", config)
    
    # 按行分析
    nan_rows = nan_mask.any(dim=1)
    nan_row_indices = torch.nonzero(nan_rows, as_tuple=False).squeeze()
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"Rows with NaN: {nan_row_indices.numel()}/{grad_cpu.shape[0]}")
        if nan_row_indices.numel() > 0:
            max_show = min(20, nan_row_indices.numel())
            if nan_row_indices.dim() == 0:
                print(f"  NaN row indices: [{nan_row_indices.item()}]")
            else:
                print(f"  First {max_show} NaN row indices: {nan_row_indices[:max_show].tolist()}")
            
            # 检查这些行是否有规律
            if nan_row_indices.numel() > 1:
                if nan_row_indices.dim() == 0:
                    diffs = torch.tensor([])
                else:
                    diffs = torch.diff(nan_row_indices.float())
                if len(diffs) > 0:
                    print(f"  Row index differences: min={diffs.min().item()}, max={diffs.max().item()}, mean={diffs.mean().item():.2f}")
    
    # 按列分析
    nan_cols = nan_mask.any(dim=0)
    nan_col_count = nan_cols.sum().item()
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"Columns with NaN: {nan_col_count}/{grad_cpu.shape[1]}")
    
    # 检查 NaN 是否集中在某些区域
    print_step("NaN Clustering Analysis", config)
    
    # 检查前几行和后几行
    first_rows = grad_cpu[:100, :]
    last_rows = grad_cpu[-100:, :]
    first_nan = torch.isnan(first_rows).sum().item()
    last_nan = torch.isnan(last_rows).sum().item()
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"NaN in first 100 rows: {first_nan}/{first_rows.numel()}")
        print(f"NaN in last 100 rows: {last_nan}/{last_rows.numel()}")
    
    # 检查 NaN 是否在特定位置
    print_step("NaN Position Analysis", config)
    nan_positions = torch.nonzero(nan_mask, as_tuple=False)
    if len(nan_positions) > 0:
        max_show = min(10, len(nan_positions))
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"First {max_show} NaN positions:")
            for i, pos in enumerate(nan_positions[:max_show]):
                print(f"  [{pos[0].item()}, {pos[1].item()}]")
        
        # 检查这些位置在期望梯度中的值
        print_step("Expected Gradient Check", config)
        grad_output = torch.ones_like(y)
        x_flat = x.view(-1, 768)
        grad_output_flat = grad_output.view(-1, 3072)
        expected_grad = torch.mm(x_flat.t(), grad_output_flat)  # (768, 3072)
        expected_grad_t = expected_grad.t()  # (3072, 768)
        
        expected_cpu = expected_grad_t.cpu()
        expected_nan = torch.isnan(expected_cpu).any().item()
        
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"Expected gradient has NaN: {expected_nan}")
        
        if not expected_nan:
            if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
                print("Checking NaN positions in expected gradient:")
                max_check = min(5, len(nan_positions))
                for i, pos in enumerate(nan_positions[:max_check]):
                    row, col = pos[0].item(), pos[1].item()
                    exp_val = expected_cpu[row, col].item()
                    print(f"  Position [{row}, {col}]: expected={exp_val:.6f}")
        
        # 检查实际梯度中非 NaN 位置的值
        valid_mask = ~nan_mask
        if valid_mask.any():
            actual_valid = grad_cpu[valid_mask]
            expected_valid = expected_cpu[valid_mask]
            if len(actual_valid) == len(expected_valid):
                diff = (actual_valid - expected_valid).abs()
                if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                    print(f"\n[Valid Values Comparison]")
                    print(f"  Valid elements: {len(actual_valid)}/{grad_cpu.numel()}")
                    print(f"  Max diff: {diff.max().item():.6f}")
                    print(f"  Mean diff: {diff.mean().item():.6f}")
    
    # 检查梯度 tensor 的内存布局
    print_step("Memory Layout Analysis", config)
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  Is contiguous: {linear.weight.grad.is_contiguous()}")
        print(f"  Strides: {linear.weight.grad.stride()}")
        print(f"  Storage offset: {linear.weight.grad.storage_offset()}")


def main():
    parser = create_arg_parser("Gradient NaN Pattern Analyzer")
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_echo_npu()
    
    try:
        analyze_gradient_nan_pattern(config)
        return True
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"✗ Analysis failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

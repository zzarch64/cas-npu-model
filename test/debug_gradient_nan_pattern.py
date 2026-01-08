#!/usr/bin/env python
"""
分析梯度NaN的分布模式
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cas_npu
    print("✓ CAS-NPU extension imported successfully")
except ImportError as e:
    print(f"✗ Failed to import CAS-NPU extension: {e}")
    sys.exit(1)

def analyze_nan_pattern():
    """分析NaN的分布模式"""
    print("=" * 80)
    print("Gradient NaN Pattern Analysis")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    # 创建Linear层
    linear = nn.Linear(768, 3072).to(device)
    x = torch.randn(2, 10, 768, device=device)
    
    # 前向传播
    y = linear(x)
    loss = y.sum()
    
    # 反向传播
    loss.backward()
    
    if linear.weight.grad is None:
        print("✗ weight.grad is None!")
        return
    
    grad_cpu = linear.weight.grad.cpu()
    nan_mask = torch.isnan(grad_cpu)
    
    print(f"\nGradient shape: {grad_cpu.shape}")
    print(f"NaN count: {nan_mask.sum().item()}/{grad_cpu.numel()}")
    
    # 分析NaN的分布
    print("\n[NaN Distribution Analysis]")
    
    # 按行分析
    nan_rows = nan_mask.any(dim=1)
    nan_row_indices = torch.nonzero(nan_rows, as_tuple=False).squeeze()
    print(f"Rows with NaN: {nan_row_indices.numel()}/{grad_cpu.shape[0]}")
    if nan_row_indices.numel() > 0:
        print(f"  First 20 NaN row indices: {nan_row_indices[:20].tolist()}")
        
        # 检查这些行是否有规律
        if nan_row_indices.numel() > 1:
            diffs = torch.diff(nan_row_indices.float())
            print(f"  Row index differences: min={diffs.min().item()}, max={diffs.max().item()}, mean={diffs.mean().item():.2f}")
    
    # 按列分析
    nan_cols = nan_mask.any(dim=0)
    nan_col_count = nan_cols.sum().item()
    print(f"Columns with NaN: {nan_col_count}/{grad_cpu.shape[1]}")
    
    # 检查NaN是否集中在某些区域
    print("\n[NaN Clustering Analysis]")
    
    # 检查前几行和后几行
    first_rows = grad_cpu[:100, :]
    last_rows = grad_cpu[-100:, :]
    first_nan = torch.isnan(first_rows).sum().item()
    last_nan = torch.isnan(last_rows).sum().item()
    print(f"NaN in first 100 rows: {first_nan}/{first_rows.numel()}")
    print(f"NaN in last 100 rows: {last_nan}/{last_rows.numel()}")
    
    # 检查NaN是否在特定位置
    print("\n[NaN Position Analysis]")
    nan_positions = torch.nonzero(nan_mask, as_tuple=False)
    if len(nan_positions) > 0:
        print(f"First 10 NaN positions:")
        for i, pos in enumerate(nan_positions[:10]):
            print(f"  [{pos[0].item()}, {pos[1].item()}]")
        
        # 检查这些位置在期望梯度中的值
        print("\n[Expected Gradient Check]")
        grad_output = torch.ones_like(y)
        x_flat = x.view(-1, 768)
        grad_output_flat = grad_output.view(-1, 3072)
        expected_grad = torch.mm(x_flat.t(), grad_output_flat)  # (768, 3072)
        expected_grad_t = expected_grad.t()  # (3072, 768)
        
        expected_cpu = expected_grad_t.cpu()
        expected_nan = torch.isnan(expected_cpu).any().item()
        print(f"Expected gradient has NaN: {expected_nan}")
        
        if not expected_nan:
            print("Checking NaN positions in expected gradient:")
            for i, pos in enumerate(nan_positions[:5]):
                row, col = pos[0].item(), pos[1].item()
                exp_val = expected_cpu[row, col].item()
                print(f"  Position [{row}, {col}]: expected={exp_val:.6f}")
        
        # 检查实际梯度中非NaN位置的值
        valid_mask = ~nan_mask
        if valid_mask.any():
            actual_valid = grad_cpu[valid_mask]
            expected_valid = expected_cpu[valid_mask]
            if len(actual_valid) == len(expected_valid):
                diff = (actual_valid - expected_valid).abs()
                print(f"\n[Valid Values Comparison]")
                print(f"  Valid elements: {len(actual_valid)}/{grad_cpu.numel()}")
                print(f"  Max diff: {diff.max().item():.6f}")
                print(f"  Mean diff: {diff.mean().item():.6f}")
    
    # 检查梯度tensor的内存布局
    print("\n[Memory Layout Analysis]")
    print(f"  Is contiguous: {linear.weight.grad.is_contiguous()}")
    print(f"  Strides: {linear.weight.grad.stride()}")
    print(f"  Storage offset: {linear.weight.grad.storage_offset()}")

if __name__ == "__main__":
    analyze_nan_pattern()

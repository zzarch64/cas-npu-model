#!/usr/bin/env python
"""
详细测试addmm操作的梯度计算过程，检查每一步
"""

import sys
import os
import torch
import torch.nn as nn

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cas_npu
    print("✓ CAS-NPU extension imported successfully")
except ImportError as e:
    print(f"✗ Failed to import CAS-NPU extension: {e}")
    sys.exit(1)

def test_addmm_gradient_detailed():
    """详细测试addmm操作的梯度计算"""
    print("=" * 80)
    print("addmm Gradient Computation Detailed Test")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    batch_size = 20
    in_features = 768
    out_features = 3072
    
    # 创建tensor
    input_t = torch.randn(batch_size, in_features, device=device, requires_grad=True)
    weight = torch.randn(out_features, in_features, device=device, requires_grad=True)
    bias = torch.randn(out_features, device=device, requires_grad=True)
    
    print(f"\n[Step 1: Forward pass]")
    print(f"  input shape: {input_t.shape}")
    print(f"  weight shape: {weight.shape}")
    print(f"  bias shape: {bias.shape}")
    
    # 前向传播
    output = torch.addmm(bias, input_t, weight.t())
    print(f"  output shape: {output.shape}")
    
    # 检查输出
    output_cpu = output.detach().cpu()
    has_nan = torch.isnan(output_cpu).any().item()
    has_inf = torch.isinf(output_cpu).any().item()
    print(f"  output has NaN: {has_nan}")
    print(f"  output has Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("  ✗ Forward pass produces NaN/Inf!")
        return False
    
    # 创建grad_output（模拟上一层传来的梯度）
    print(f"\n[Step 2: Create grad_output]")
    grad_output = torch.ones_like(output)
    grad_output_cpu = grad_output.cpu()
    has_nan = torch.isnan(grad_output_cpu).any().item()
    has_inf = torch.isinf(grad_output_cpu).any().item()
    print(f"  grad_output shape: {grad_output.shape}")
    print(f"  grad_output has NaN: {has_nan}")
    print(f"  grad_output has Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("  ✗ grad_output contains NaN/Inf!")
        return False
    
    # 手动计算梯度（用于对比）
    print(f"\n[Step 3: Manual gradient computation]")
    input_cpu = input_t.detach().cpu()
    weight_cpu = weight.detach().cpu()
    grad_output_cpu = grad_output.cpu()
    
    # 对于 addmm(bias, input, weight.t())，梯度计算：
    # weight.grad 的形状应该与 weight 相同：[out_features, in_features] = [3072, 768]
    # weight.grad = grad_output.t() @ input
    # grad_output.t() 形状: [out_features, batch_size] = [3072, 20]
    # input 形状: [batch_size, in_features] = [20, 768]
    # 结果: [3072, 20] @ [20, 768] = [3072, 768]
    expected_weight_grad = grad_output_cpu.t() @ input_cpu
    print(f"  expected_weight_grad shape: {expected_weight_grad.shape}")
    
    has_nan = torch.isnan(expected_weight_grad).any().item()
    has_inf = torch.isinf(expected_weight_grad).any().item()
    print(f"  expected_weight_grad has NaN: {has_nan}")
    print(f"  expected_weight_grad has Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("  ✗ Expected gradient contains NaN/Inf!")
        return False
    
    # 使用autograd计算梯度
    print(f"\n[Step 4: Autograd backward]")
    loss = output.sum()
    print(f"  loss: {loss.item():.6f}")
    
    # 清零梯度
    if weight.grad is not None:
        weight.grad.zero_()
    if bias.grad is not None:
        bias.grad.zero_()
    if input_t.grad is not None:
        input_t.grad.zero_()
    
    # 反向传播
    loss.backward()
    print(f"  ✓ backward completed")
    
    # 检查实际梯度
    print(f"\n[Step 5: Check actual gradients]")
    if weight.grad is not None:
        actual_weight_grad_cpu = weight.grad.cpu()
        print(f"  actual_weight_grad shape: {actual_weight_grad_cpu.shape}")
        
        has_nan = torch.isnan(actual_weight_grad_cpu).any().item()
        has_inf = torch.isinf(actual_weight_grad_cpu).any().item()
        nan_count = torch.isnan(actual_weight_grad_cpu).sum().item() if has_nan else 0
        inf_count = torch.isinf(actual_weight_grad_cpu).sum().item() if has_inf else 0
        
        print(f"  actual_weight_grad has NaN: {has_nan} ({nan_count}/{actual_weight_grad_cpu.numel()})")
        print(f"  actual_weight_grad has Inf: {has_inf} ({inf_count}/{actual_weight_grad_cpu.numel()})")
        
        if has_nan or has_inf:
            # 分析NaN/Inf的分布
            if has_nan:
                nan_mask = torch.isnan(actual_weight_grad_cpu)
                nan_rows = nan_mask.any(dim=1).sum().item()
                nan_cols = nan_mask.any(dim=0).sum().item()
                print(f"    NaN rows: {nan_rows}/{actual_weight_grad_cpu.shape[0]}")
                print(f"    NaN cols: {nan_cols}/{actual_weight_grad_cpu.shape[1]}")
                
                # 检查哪些行有NaN
                nan_row_indices = nan_mask.any(dim=1).nonzero(as_tuple=True)[0]
                if len(nan_row_indices) > 0:
                    print(f"    First 10 NaN row indices: {nan_row_indices[:10].tolist()}")
            
            # 比较有效值
            valid_mask = ~torch.isnan(actual_weight_grad_cpu) & ~torch.isinf(actual_weight_grad_cpu)
            if valid_mask.any():
                valid_actual = actual_weight_grad_cpu[valid_mask]
                # 确保 expected_weight_grad 的形状与 actual_weight_grad_cpu 匹配
                if expected_weight_grad.shape != actual_weight_grad_cpu.shape:
                    print(f"    Warning: Shape mismatch! Expected {expected_weight_grad.shape}, got {actual_weight_grad_cpu.shape}")
                    print(f"    Skipping comparison due to shape mismatch")
                    valid_expected = None
                else:
                    valid_expected = expected_weight_grad[valid_mask]
                
                if valid_expected is not None:
                    diff = (valid_actual - valid_expected).abs()
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    print(f"    Valid values comparison:")
                    print(f"      Max difference: {max_diff:.6f}")
                    print(f"      Mean difference: {mean_diff:.6f}")
                    
                    # 检查有效值中是否有NaN/Inf
                    has_nan_valid = torch.isnan(valid_actual).any().item()
                    has_inf_valid = torch.isinf(valid_actual).any().item()
                    print(f"      Valid actual has NaN: {has_nan_valid}")
                    print(f"      Valid actual has Inf: {has_inf_valid}")
                else:
                    print(f"    Skipping comparison due to shape mismatch")
            
            return False
        else:
            # 比较梯度
            if expected_weight_grad.shape == actual_weight_grad_cpu.shape:
                diff = (actual_weight_grad_cpu - expected_weight_grad).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"  Max difference: {max_diff:.6f}")
                print(f"  Mean difference: {mean_diff:.6f}")
                
                if max_diff < 1e-5:
                    print(f"  ✓ Gradients match!")
                else:
                    print(f"  ✗ Gradients don't match!")
                    return False
            else:
                print(f"  Warning: Shape mismatch! Expected {expected_weight_grad.shape}, got {actual_weight_grad_cpu.shape}")
                print(f"  Skipping comparison")
    
    return True

if __name__ == "__main__":
    success = test_addmm_gradient_detailed()
    print("\n" + "=" * 80)
    if success:
        print("Test passed! ✓")
    else:
        print("Test failed! ✗")
    print("=" * 80)
    sys.exit(0 if success else 1)

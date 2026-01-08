#!/usr/bin/env python
"""
逐步测试addmm操作的梯度计算过程，检查每一步的tensor状态
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

def check_tensor(tensor, name, device=None):
    """检查tensor的状态"""
    if tensor is None:
        print(f"  {name}: None")
        return
    
    if device is not None:
        tensor_cpu = tensor.cpu() if tensor.device != torch.device('cpu') else tensor
    else:
        tensor_cpu = tensor.cpu() if hasattr(tensor, 'device') and tensor.device != torch.device('cpu') else tensor
    
    has_nan = torch.isnan(tensor_cpu).any().item()
    has_inf = torch.isinf(tensor_cpu).any().item()
    nan_count = torch.isnan(tensor_cpu).sum().item() if has_nan else 0
    inf_count = torch.isinf(tensor_cpu).sum().item() if has_inf else 0
    
    print(f"  {name}:")
    print(f"    shape: {tensor.shape}")
    print(f"    device: {tensor.device}")
    print(f"    has NaN: {has_nan} ({nan_count}/{tensor_cpu.numel()})")
    print(f"    has Inf: {has_inf} ({inf_count}/{tensor_cpu.numel()})")
    
    if not has_nan and not has_inf:
        print(f"    min: {tensor_cpu.min().item():.6f}")
        print(f"    max: {tensor_cpu.max().item():.6f}")
        print(f"    mean: {tensor_cpu.mean().item():.6f}")

def test_addmm_gradient_step_by_step():
    """逐步测试addmm操作的梯度计算"""
    print("=" * 80)
    print("addmm Gradient Computation Step-by-Step Test")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    batch_size = 20
    in_features = 768
    out_features = 3072
    
    # 创建tensor
    print(f"\n[Step 1: Create input tensors]")
    input_t = torch.randn(batch_size, in_features, device=device, requires_grad=True)
    weight = torch.randn(out_features, in_features, device=device, requires_grad=True)
    bias = torch.randn(out_features, device=device, requires_grad=True)
    
    check_tensor(input_t, "input_t", device)
    check_tensor(weight, "weight", device)
    check_tensor(bias, "bias", device)
    
    # 前向传播
    print(f"\n[Step 2: Forward pass]")
    output = torch.addmm(bias, input_t, weight.t())
    check_tensor(output, "output", device)
    
    if torch.isnan(output.cpu()).any() or torch.isinf(output.cpu()).any():
        print("  ✗ Forward pass produces NaN/Inf!")
        return False
    
    # 创建grad_output
    print(f"\n[Step 3: Create grad_output]")
    grad_output = torch.ones_like(output)
    check_tensor(grad_output, "grad_output", device)
    
    # 手动计算期望的梯度（用于对比）
    print(f"\n[Step 4: Manual gradient computation]")
    input_cpu = input_t.detach().cpu()
    weight_cpu = weight.detach().cpu()
    grad_output_cpu = grad_output.cpu()
    
    # weight.grad = grad_output.t() @ input
    expected_weight_grad = grad_output_cpu.t() @ input_cpu
    check_tensor(expected_weight_grad, "expected_weight_grad")
    
    if torch.isnan(expected_weight_grad).any() or torch.isinf(expected_weight_grad).any():
        print("  ✗ Expected gradient contains NaN/Inf!")
        return False
    
    # 使用autograd计算梯度
    print(f"\n[Step 5: Autograd backward]")
    loss = output.sum()
    print(f"  loss: {loss.item():.6f}")
    
    # 清零梯度
    if weight.grad is not None:
        weight.grad.zero_()
        print(f"  Cleared weight.grad")
    if bias.grad is not None:
        bias.grad.zero_()
        print(f"  Cleared bias.grad")
    if input_t.grad is not None:
        input_t.grad.zero_()
        print(f"  Cleared input_t.grad")
    
    # 反向传播
    print(f"  Executing loss.backward()...")
    loss.backward()
    print(f"  ✓ backward completed")
    
    # 检查实际梯度
    print(f"\n[Step 6: Check actual gradients]")
    if weight.grad is not None:
        check_tensor(weight.grad, "weight.grad", device)
        
        actual_weight_grad_cpu = weight.grad.cpu()
        has_nan = torch.isnan(actual_weight_grad_cpu).any().item()
        has_inf = torch.isinf(actual_weight_grad_cpu).any().item()
        
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
                    
                    # 检查这些行的数据
                    for idx in nan_row_indices[:3]:
                        row = actual_weight_grad_cpu[idx]
                        nan_in_row = torch.isnan(row).sum().item()
                        print(f"      Row {idx}: {nan_in_row}/{row.numel()} NaN")
            
            # 比较有效值
            valid_mask = ~torch.isnan(actual_weight_grad_cpu) & ~torch.isinf(actual_weight_grad_cpu)
            if valid_mask.any():
                valid_actual = actual_weight_grad_cpu[valid_mask]
                valid_expected = expected_weight_grad[valid_mask]
                
                diff = (valid_actual - valid_expected).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"    Valid values comparison:")
                print(f"      Valid count: {valid_mask.sum().item()}/{actual_weight_grad_cpu.numel()}")
                print(f"      Max difference: {max_diff:.6f}")
                print(f"      Mean difference: {mean_diff:.6f}")
                
                # 检查有效值中是否有NaN/Inf
                has_nan_valid = torch.isnan(valid_actual).any().item()
                has_inf_valid = torch.isinf(valid_actual).any().item()
                print(f"      Valid actual has NaN: {has_nan_valid}")
                print(f"      Valid actual has Inf: {has_inf_valid}")
                
                # 检查有效值的范围
                if not has_nan_valid and not has_inf_valid:
                    print(f"      Valid actual range: [{valid_actual.min().item():.6f}, {valid_actual.max().item():.6f}]")
                    print(f"      Valid expected range: [{valid_expected.min().item():.6f}, {valid_expected.max().item():.6f}]")
            
            return False
        else:
            # 比较梯度
            if expected_weight_grad.shape == actual_weight_grad_cpu.shape:
                diff = (actual_weight_grad_cpu - expected_weight_grad).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"    Gradient comparison:")
                print(f"      Max difference: {max_diff:.6f}")
                print(f"      Mean difference: {mean_diff:.6f}")
                
                if max_diff < 1e-5:
                    print(f"    ✓ Gradients match!")
                    return True
                else:
                    print(f"    ✗ Gradients don't match!")
                    return False
            else:
                print(f"    Warning: Shape mismatch! Expected {expected_weight_grad.shape}, got {actual_weight_grad_cpu.shape}")
                return False
    
    return True

if __name__ == "__main__":
    success = test_addmm_gradient_step_by_step()
    print("\n" + "=" * 80)
    if success:
        print("Test passed! ✓")
    else:
        print("Test failed! ✗")
    print("=" * 80)
    sys.exit(0 if success else 1)

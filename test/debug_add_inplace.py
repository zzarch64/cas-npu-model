#!/usr/bin/env python
"""
测试add_.Tensor操作（梯度累积时使用）
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

def test_add_inplace():
    """测试add_.Tensor操作"""
    print("=" * 80)
    print("add_.Tensor Operation Test")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    # 创建两个tensor
    a = torch.randn(3072, 768, device=device)
    b = torch.randn(3072, 768, device=device)
    
    print(f"\n[Initial State]")
    print(f"  a shape: {a.shape}, device: {a.device}")
    print(f"  b shape: {b.shape}, device: {b.device}")
    
    # 检查初始状态
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    a_has_nan = torch.isnan(a_cpu).any().item()
    b_has_nan = torch.isnan(b_cpu).any().item()
    print(f"  a has NaN: {a_has_nan}")
    print(f"  b has NaN: {b_has_nan}")
    
    # 测试add_.Tensor
    print(f"\n[Testing add_.Tensor]")
    print(f"  Executing a.add_(b)...")
    
    try:
        a.add_(b)
        
        print(f"  ✓ add_.Tensor completed")
        
        # 检查结果
        a_cpu_after = a.cpu()
        has_nan = torch.isnan(a_cpu_after).any().item()
        has_inf = torch.isinf(a_cpu_after).any().item()
        
        print(f"  Result has NaN: {has_nan}")
        print(f"  Result has Inf: {has_inf}")
        
        if has_nan:
            nan_count = torch.isnan(a_cpu_after).sum().item()
            print(f"    NaN count: {nan_count}/{a_cpu_after.numel()}")
        if has_inf:
            inf_count = torch.isinf(a_cpu_after).sum().item()
            print(f"    Inf count: {inf_count}/{a_cpu_after.numel()}")
        
        # 验证结果
        expected = a_cpu + b_cpu
        if not torch.isnan(a_cpu_after).any():
            diff = (a_cpu_after - expected).abs()
            max_diff = diff.max().item()
            print(f"  Max difference from expected: {max_diff:.6f}")
            if max_diff < 1e-5:
                print(f"  ✓ Result matches expected")
            else:
                print(f"  ✗ Result doesn't match expected")
        else:
            print(f"  ✗ Result contains NaN, cannot verify")
        
        return not (has_nan or has_inf)
        
    except Exception as e:
        print(f"  ✗ add_.Tensor failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_accumulation_simulation():
    """模拟梯度累积过程"""
    print("\n" + "=" * 80)
    print("Gradient Accumulation Simulation")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    # 模拟梯度tensor（初始化为0）
    grad = torch.zeros(3072, 768, device=device)
    print(f"\n[Initial gradient tensor]")
    print(f"  Shape: {grad.shape}")
    print(f"  Device: {grad.device}")
    print(f"  Is contiguous: {grad.is_contiguous()}")
    
    # 检查初始状态
    grad_cpu = grad.cpu()
    has_nan = torch.isnan(grad_cpu).any().item()
    has_inf = torch.isinf(grad_cpu).any().item()
    zero_count = (grad_cpu == 0).sum().item()
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Zero count: {zero_count}/{grad_cpu.numel()}")
    
    # 模拟梯度累积（添加新的梯度）
    new_grad = torch.randn(3072, 768, device=device)
    print(f"\n[Adding new gradient]")
    print(f"  New grad shape: {new_grad.shape}")
    
    new_grad_cpu = new_grad.cpu()
    new_has_nan = torch.isnan(new_grad_cpu).any().item()
    print(f"  New grad has NaN: {new_has_nan}")
    
    # 使用add_累积梯度
    print(f"\n[Accumulating gradient]")
    print(f"  Executing grad.add_(new_grad)...")
    
    try:
        grad.add_(new_grad)
        
        # 检查结果
        grad_cpu_after = grad.cpu()
        has_nan_after = torch.isnan(grad_cpu_after).any().item()
        has_inf_after = torch.isinf(grad_cpu_after).any().item()
        
        print(f"  Result has NaN: {has_nan_after}")
        print(f"  Result has Inf: {has_inf_after}")
        
        if has_nan_after:
            nan_count = torch.isnan(grad_cpu_after).sum().item()
            nan_rows = torch.isnan(grad_cpu_after).any(dim=1).sum().item()
            print(f"    NaN count: {nan_count}/{grad_cpu_after.numel()}")
            print(f"    NaN rows: {nan_rows}/{grad_cpu_after.shape[0]}")
        
        # 验证结果
        expected = grad_cpu + new_grad_cpu
        if not torch.isnan(grad_cpu_after).any():
            diff = (grad_cpu_after - expected).abs()
            max_diff = diff.max().item()
            print(f"  Max difference from expected: {max_diff:.6f}")
            if max_diff < 1e-5:
                print(f"  ✓ Result matches expected")
            else:
                print(f"  ✗ Result doesn't match expected")
        else:
            print(f"  ✗ Result contains NaN, cannot verify")
        
        return not (has_nan_after or has_inf_after)
        
    except Exception as e:
        print(f"  ✗ Gradient accumulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_add_inplace()
    success2 = test_gradient_accumulation_simulation()
    
    print("\n" + "=" * 80)
    if success1 and success2:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 80)
    
    sys.exit(0 if (success1 and success2) else 1)

#!/usr/bin/env python
"""
专门用于debug梯度计算问题的脚本
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

def check_gradient_creation():
    """检查梯度是如何被创建的"""
    print("=" * 80)
    print("Gradient Creation Debug")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    # 创建一个简单的Linear层
    linear = nn.Linear(10, 5).to(device)
    x = torch.randn(2, 10, device=device, requires_grad=True)
    
    print("\n[Step 1] Initial state:")
    print(f"  linear.weight.requires_grad: {linear.weight.requires_grad}")
    print(f"  linear.weight.grad: {linear.weight.grad}")
    print(f"  x.requires_grad: {x.requires_grad}")
    
    # 前向传播
    print("\n[Step 2] Forward pass:")
    y = linear(x)
    print(f"  y.shape: {y.shape}")
    print(f"  y.device: {y.device}")
    print(f"  y.requires_grad: {y.requires_grad}")
    print(f"  y.grad_fn: {y.grad_fn}")
    if y.grad_fn:
        print(f"    grad_fn type: {type(y.grad_fn).__name__}")
        print(f"    next_functions: {y.grad_fn.next_functions}")
    
    # 计算损失
    loss = y.sum()
    print(f"\n[Step 3] Loss:")
    print(f"  loss: {loss.item():.6f}")
    print(f"  loss.grad_fn: {loss.grad_fn}")
    if loss.grad_fn:
        print(f"    grad_fn type: {type(loss.grad_fn).__name__}")
    
    # 反向传播
    print(f"\n[Step 4] Backward pass:")
    print("  Executing loss.backward()...")
    
    try:
        loss.backward()
        
        print(f"  ✓ backward() completed")
        
        # 检查梯度
        print(f"\n[Step 5] Checking gradients:")
        
        if linear.weight.grad is not None:
            print(f"  Weight grad exists: ✓")
            print(f"    Device: {linear.weight.grad.device}")
            print(f"    Shape: {linear.weight.grad.shape}")
            print(f"    Dtype: {linear.weight.grad.dtype}")
            
            # 检查是否包含NaN/Inf
            cpu_grad = linear.weight.grad.cpu()
            has_nan = torch.isnan(cpu_grad).any().item()
            has_inf = torch.isinf(cpu_grad).any().item()
            
            if has_nan:
                nan_count = torch.isnan(cpu_grad).sum().item()
                print(f"    ✗ Contains NaN: {nan_count}/{cpu_grad.numel()}")
                print(f"    Min: {cpu_grad.min().item()}, Max: {cpu_grad.max().item()}")
            elif has_inf:
                inf_count = torch.isinf(cpu_grad).sum().item()
                print(f"    ✗ Contains Inf: {inf_count}/{cpu_grad.numel()}")
                print(f"    Min: {cpu_grad.min().item()}, Max: {cpu_grad.max().item()}")
            else:
                print(f"    ✓ No NaN/Inf")
                print(f"    Stats: min={cpu_grad.min().item():.6f}, max={cpu_grad.max().item():.6f}, mean={cpu_grad.mean().item():.6f}")
        else:
            print(f"  ✗ Weight grad is None!")
        
        if linear.bias.grad is not None:
            print(f"  Bias grad exists: ✓")
            print(f"    Device: {linear.bias.grad.device}")
            print(f"    Shape: {linear.bias.grad.shape}")
            
            cpu_grad = linear.bias.grad.cpu()
            has_nan = torch.isnan(cpu_grad).any().item()
            has_inf = torch.isinf(cpu_grad).any().item()
            
            if has_nan or has_inf:
                print(f"    ✗ Contains NaN/Inf")
            else:
                print(f"    ✓ No NaN/Inf")
                print(f"    Stats: min={cpu_grad.min().item():.6f}, max={cpu_grad.max().item():.6f}, mean={cpu_grad.mean().item():.6f}")
        else:
            print(f"  ✗ Bias grad is None!")
        
        if x.grad is not None:
            print(f"  Input grad exists: ✓")
            print(f"    Device: {x.grad.device}")
            print(f"    Shape: {x.grad.shape}")
            
            cpu_grad = x.grad.cpu()
            has_nan = torch.isnan(cpu_grad).any().item()
            has_inf = torch.isinf(cpu_grad).any().item()
            
            if has_nan or has_inf:
                print(f"    ✗ Contains NaN/Inf")
            else:
                print(f"    ✓ No NaN/Inf")
        else:
            print(f"  Input grad: None (this is OK for parameters)")
        
        # 手动验证梯度计算
        print(f"\n[Step 6] Manual gradient verification:")
        print("  Computing weight.grad manually...")
        
        # weight.grad应该等于 input.t() @ grad_output
        # 对于Linear层: y = x @ W.t() + b
        # weight.grad = x.t() @ grad_output (where grad_output = ones)
        grad_output = torch.ones_like(y)
        expected_weight_grad = torch.mm(x.t(), grad_output)
        
        print(f"    Expected weight grad shape: {expected_weight_grad.shape}")
        print(f"    Actual weight grad shape: {linear.weight.grad.shape if linear.weight.grad is not None else 'None'}")
        
        if linear.weight.grad is not None:
            expected_cpu = expected_weight_grad.cpu()
            actual_cpu = linear.weight.grad.cpu()
            
            # 检查是否匹配（忽略NaN）
            if not torch.isnan(actual_cpu).any():
                diff = (expected_cpu - actual_cpu).abs()
                max_diff = diff.max().item()
                print(f"    Max difference: {max_diff:.6f}")
                if max_diff < 1e-5:
                    print(f"    ✓ Gradients match!")
                else:
                    print(f"    ✗ Gradients don't match!")
            else:
                print(f"    ✗ Actual gradient contains NaN, cannot compare")
        
        return True
        
    except Exception as e:
        print(f"  ✗ backward() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_gradient_creation()
    sys.exit(0 if success else 1)

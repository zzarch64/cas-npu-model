#!/usr/bin/env python
"""
调试梯度tensor创建过程
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

def test_gradient_tensor_creation():
    """测试梯度tensor的创建过程"""
    print("=" * 80)
    print("Gradient Tensor Creation Debug")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    # 创建Linear层
    linear = nn.Linear(768, 3072).to(device)
    x = torch.randn(2, 10, 768, device=device)
    
    print("\n[Step 1] Before forward pass:")
    print(f"  weight.grad: {linear.weight.grad}")
    print(f"  bias.grad: {linear.bias.grad}")
    
    # 前向传播
    y = linear(x)
    loss = y.sum()
    
    print("\n[Step 2] After forward pass, before backward:")
    print(f"  weight.grad: {linear.weight.grad}")
    print(f"  bias.grad: {linear.bias.grad}")
    
    # 手动创建梯度tensor，看看是否有问题
    print("\n[Step 3] Manually creating gradient tensors:")
    try:
        # 使用zeros_like创建梯度tensor
        manual_weight_grad = torch.zeros_like(linear.weight)
        print(f"  Manual weight grad shape: {manual_weight_grad.shape}")
        print(f"  Manual weight grad device: {manual_weight_grad.device}")
        
        # 检查是否包含NaN
        cpu_grad = manual_weight_grad.cpu()
        has_nan = torch.isnan(cpu_grad).any().item()
        has_inf = torch.isinf(cpu_grad).any().item()
        zero_count = (cpu_grad == 0).sum().item()
        
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        print(f"  Zero count: {zero_count}/{cpu_grad.numel()} ({zero_count/cpu_grad.numel()*100:.2f}%)")
        
        if has_nan or has_inf:
            print("  ✗ Manually created gradient tensor contains NaN/Inf!")
            return False
        else:
            print("  ✓ Manually created gradient tensor is OK")
    except Exception as e:
        print(f"  ✗ Failed to create manual gradient tensor: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 执行反向传播
    print("\n[Step 4] Executing backward pass...")
    try:
        loss.backward()
        
        print("\n[Step 5] After backward pass:")
        if linear.weight.grad is not None:
            print(f"  weight.grad shape: {linear.weight.grad.shape}")
            print(f"  weight.grad device: {linear.weight.grad.device}")
            
            cpu_grad = linear.weight.grad.cpu()
            has_nan = torch.isnan(cpu_grad).any().item()
            has_inf = torch.isinf(cpu_grad).any().item()
            zero_count = (cpu_grad == 0).sum().item()
            nan_count = torch.isnan(cpu_grad).sum().item()
            inf_count = torch.isinf(cpu_grad).sum().item()
            
            print(f"  Has NaN: {has_nan} ({nan_count}/{cpu_grad.numel()} = {nan_count/cpu_grad.numel()*100:.2f}%)")
            print(f"  Has Inf: {has_inf} ({inf_count}/{cpu_grad.numel()} = {inf_count/cpu_grad.numel()*100:.2f}%)")
            print(f"  Zero count: {zero_count}/{cpu_grad.numel()} ({zero_count/cpu_grad.numel()*100:.2f}%)")
            
            # 检查NaN的分布
            if has_nan:
                nan_mask = torch.isnan(cpu_grad)
                nan_rows = nan_mask.any(dim=1).sum().item()
                nan_cols = nan_mask.any(dim=0).sum().item()
                print(f"  NaN rows: {nan_rows}/{cpu_grad.shape[0]}")
                print(f"  NaN cols: {nan_cols}/{cpu_grad.shape[1]}")
                
                # 检查NaN是否在特定位置
                nan_indices = torch.nonzero(nan_mask, as_tuple=False)
                print(f"  First 10 NaN positions:")
                for i, idx in enumerate(nan_indices[:10]):
                    print(f"    [{idx[0].item()}, {idx[1].item()}]")
        else:
            print("  ✗ weight.grad is None!")
            return False
        
        # 比较手动创建的梯度tensor和实际梯度tensor
        print("\n[Step 6] Comparing manual and actual gradient tensors:")
        if linear.weight.grad is not None and manual_weight_grad is not None:
            # 检查它们是否是同一个tensor
            print(f"  Same tensor object: {linear.weight.grad is manual_weight_grad}")
            print(f"  Same data pointer: {linear.weight.grad.data_ptr() == manual_weight_grad.data_ptr()}")
            
            # 检查实际梯度tensor中哪些位置是NaN，哪些是0
            actual_cpu = linear.weight.grad.cpu()
            manual_cpu = manual_weight_grad.cpu()
            
            # 找出实际梯度中NaN的位置，看看手动创建的梯度在这些位置是什么
            if torch.isnan(actual_cpu).any():
                nan_mask = torch.isnan(actual_cpu)
                nan_positions = torch.nonzero(nan_mask, as_tuple=False)
                print(f"  Checking NaN positions in manual grad:")
                for i, idx in enumerate(nan_positions[:5]):
                    row, col = idx[0].item(), idx[1].item()
                    manual_val = manual_cpu[row, col].item()
                    print(f"    Position [{row}, {col}]: manual={manual_val:.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gradient_tensor_creation()
    sys.exit(0 if success else 1)

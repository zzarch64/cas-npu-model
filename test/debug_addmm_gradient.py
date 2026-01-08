#!/usr/bin/env python
"""
测试addmm操作在梯度计算中的行为
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

def test_addmm_gradient():
    """测试addmm操作的梯度计算"""
    print("=" * 80)
    print("addmm Gradient Computation Test")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    # 创建tensor（模拟Linear层的前向传播）
    # Linear层使用: output = input @ weight.t() + bias
    # 这等价于: output = addmm(bias, input, weight.t())
    batch_size = 20
    in_features = 768
    out_features = 3072
    
    input_t = torch.randn(batch_size, in_features, device=device, requires_grad=True)
    weight = torch.randn(out_features, in_features, device=device, requires_grad=True)
    bias = torch.randn(out_features, device=device, requires_grad=True)
    
    print(f"\n[Input tensors]")
    print(f"  input shape: {input_t.shape}")
    print(f"  weight shape: {weight.shape}")
    print(f"  bias shape: {bias.shape}")
    
    # 检查初始状态
    input_cpu = input_t.cpu()
    weight_cpu = weight.cpu()
    bias_cpu = bias.cpu()
    
    print(f"  input has NaN: {torch.isnan(input_cpu).any().item()}")
    print(f"  weight has NaN: {torch.isnan(weight_cpu).any().item()}")
    print(f"  bias has NaN: {torch.isnan(bias_cpu).any().item()}")
    
    # 使用addmm（Linear层内部使用这个）
    print(f"\n[Forward pass with addmm]")
    print(f"  Executing: output = addmm(bias, input, weight.t())...")
    
    try:
        output = torch.addmm(bias, input_t, weight.t())
        print(f"  ✓ addmm completed")
        print(f"  output shape: {output.shape}")
        
        output_cpu = output.cpu()
        has_nan = torch.isnan(output_cpu).any().item()
        has_inf = torch.isinf(output_cpu).any().item()
        print(f"  output has NaN: {has_nan}")
        print(f"  output has Inf: {has_inf}")
        
        if has_nan or has_inf:
            print("  ✗ Forward pass produces NaN/Inf!")
            return False
        
        # 计算损失并反向传播
        print(f"\n[Backward pass]")
        loss = output.sum()
        print(f"  loss: {loss.item():.6f}")
        
        print(f"  Executing loss.backward()...")
        loss.backward()
        
        print(f"  ✓ backward completed")
        
        # 检查梯度
        print(f"\n[Gradient check]")
        if weight.grad is not None:
            grad_cpu = weight.grad.cpu()
            has_nan = torch.isnan(grad_cpu).any().item()
            has_inf = torch.isinf(grad_cpu).any().item()
            nan_count = torch.isnan(grad_cpu).sum().item() if has_nan else 0
            inf_count = torch.isinf(grad_cpu).sum().item() if has_inf else 0
            
            print(f"  weight.grad shape: {weight.grad.shape}")
            print(f"  weight.grad has NaN: {has_nan} ({nan_count}/{grad_cpu.numel()})")
            print(f"  weight.grad has Inf: {has_inf} ({inf_count}/{grad_cpu.numel()})")
            
            if has_nan or has_inf:
                # 分析NaN/Inf的分布
                if has_nan:
                    nan_mask = torch.isnan(grad_cpu)
                    nan_rows = nan_mask.any(dim=1).sum().item()
                    nan_cols = nan_mask.any(dim=0).sum().item()
                    print(f"    NaN rows: {nan_rows}/{grad_cpu.shape[0]}")
                    print(f"    NaN cols: {nan_cols}/{grad_cpu.shape[1]}")
                
                return False
            else:
                print(f"  ✓ weight.grad is OK")
        
        if bias.grad is not None:
            grad_cpu = bias.grad.cpu()
            has_nan = torch.isnan(grad_cpu).any().item()
            has_inf = torch.isinf(grad_cpu).any().item()
            print(f"  bias.grad shape: {bias.grad.shape}")
            print(f"  bias.grad has NaN: {has_nan}")
            print(f"  bias.grad has Inf: {has_inf}")
            
            if has_nan or has_inf:
                return False
            else:
                print(f"  ✓ bias.grad is OK")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_addmm_gradient()
    print("\n" + "=" * 80)
    if success:
        print("Test passed! ✓")
    else:
        print("Test failed! ✗")
    print("=" * 80)
    sys.exit(0 if success else 1)

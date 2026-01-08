#!/usr/bin/env python
"""
专门用于debug Linear Layer问题的脚本
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

def detailed_check(tensor, name):
    """详细检查tensor"""
    print(f"\n=== Checking {name} ===")
    print(f"  Shape: {tensor.shape}")
    print(f"  Device: {tensor.device}")
    print(f"  Dtype: {tensor.dtype}")
    
    cpu_tensor = tensor.cpu() if tensor.device.type == 'cas_npu' else tensor
    
    has_nan = torch.isnan(cpu_tensor).any().item()
    has_inf = torch.isinf(cpu_tensor).any().item()
    
    if has_nan:
        nan_count = torch.isnan(cpu_tensor).sum().item()
        print(f"  ✗ Contains NaN: {nan_count}/{cpu_tensor.numel()} ({nan_count/cpu_tensor.numel()*100:.2f}%)")
        nan_indices = torch.nonzero(torch.isnan(cpu_tensor), as_tuple=False)
        if len(nan_indices) > 0:
            print(f"    First NaN at index: {nan_indices[0].tolist()}")
    else:
        print(f"  ✓ No NaN")
    
    if has_inf:
        inf_count = torch.isinf(cpu_tensor).sum().item()
        print(f"  ✗ Contains Inf: {inf_count}/{cpu_tensor.numel()} ({inf_count/cpu_tensor.numel()*100:.2f}%)")
    else:
        print(f"  ✓ No Inf")
    
    if not has_nan and not has_inf:
        print(f"  Stats: min={cpu_tensor.min().item():.6f}, max={cpu_tensor.max().item():.6f}, mean={cpu_tensor.mean().item():.6f}, std={cpu_tensor.std().item():.6f}")
    
    return has_nan or has_inf

def test_linear_step_by_step():
    """逐步测试Linear层"""
    print("=" * 80)
    print("Linear Layer Step-by-Step Debug")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    # Step 1: 创建Linear层
    print("\n[Step 1] Creating Linear layer...")
    linear = nn.Linear(768, 3072).to(device)
    print(f"  Linear layer created: {linear}")
    
    # Step 2: 检查权重和偏置
    print("\n[Step 2] Checking weight and bias...")
    weight_has_issue = detailed_check(linear.weight, "Weight")
    bias_has_issue = detailed_check(linear.bias, "Bias")
    
    if weight_has_issue or bias_has_issue:
        print("\n✗ Weight or Bias has NaN/Inf - this is the problem!")
        return False
    
    # Step 3: 创建输入
    print("\n[Step 3] Creating input tensor...")
    x = torch.randn(2, 10, 768, device=device)
    input_has_issue = detailed_check(x, "Input")
    
    if input_has_issue:
        print("\n✗ Input has NaN/Inf - this is the problem!")
        return False
    
    # Step 4: 手动执行矩阵乘法（检查mm操作）
    print("\n[Step 4] Testing mm operation manually...")
    x_flat = x.view(-1, 768)  # (20, 768)
    weight_t = linear.weight.t()  # (768, 3072)
    
    print(f"  x_flat shape: {x_flat.shape}")
    print(f"  weight_t shape: {weight_t.shape}")
    
    detailed_check(x_flat, "x_flat (input for mm)")
    detailed_check(weight_t, "weight_t (weight for mm)")
    
    print("\n  Executing mm(x_flat, weight_t)...")
    try:
        mm_result = torch.mm(x_flat, weight_t)
        mm_has_issue = detailed_check(mm_result, "mm result")
        
        if mm_has_issue:
            print("\n✗ mm operation produces NaN/Inf!")
            print("  This suggests the mm implementation has a problem.")
            return False
        else:
            print("  ✓ mm operation OK")
    except Exception as e:
        print(f"\n✗ mm operation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: 添加偏置
    print("\n[Step 5] Adding bias...")
    try:
        mm_plus_bias = mm_result + linear.bias.unsqueeze(0)
        bias_add_has_issue = detailed_check(mm_plus_bias, "mm_result + bias")
        
        if bias_add_has_issue:
            print("\n✗ Adding bias produces NaN/Inf!")
            return False
        else:
            print("  ✓ Adding bias OK")
    except Exception as e:
        print(f"\n✗ Adding bias failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: 使用Linear层的前向传播
    print("\n[Step 6] Using Linear layer forward pass...")
    try:
        y = linear(x)
        linear_has_issue = detailed_check(y, "Linear output")
        
        if linear_has_issue:
            print("\n✗ Linear forward pass produces NaN/Inf!")
            print("  But manual mm + bias was OK - this suggests Linear layer wrapper has a problem.")
            return False
        else:
            print("  ✓ Linear forward pass OK")
    except Exception as e:
        print(f"\n✗ Linear forward pass failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 7: 反向传播
    print("\n[Step 7] Testing backward pass...")
    try:
        loss = y.sum()
        print(f"  Loss: {loss.item():.6f}")
        
        # 检查loss的grad_fn
        print(f"  Loss grad_fn: {loss.grad_fn}")
        if loss.grad_fn:
            print(f"    grad_fn type: {type(loss.grad_fn).__name__}")
            print(f"    next_functions: {loss.grad_fn.next_functions}")
        
        # 检查y的grad_fn
        print(f"  y grad_fn: {y.grad_fn}")
        if y.grad_fn:
            print(f"    grad_fn type: {type(y.grad_fn).__name__}")
        
        # 检查weight和bias的requires_grad状态
        print(f"  Weight requires_grad: {linear.weight.requires_grad}")
        print(f"  Bias requires_grad: {linear.bias.requires_grad}")
        
        # 检查grad是否已经存在（可能未初始化）
        print(f"  Weight grad before backward: {linear.weight.grad}")
        print(f"  Bias grad before backward: {linear.bias.grad}")
        
        # 执行反向传播
        print("\n  Executing loss.backward()...")
        
        # 在反向传播前，手动计算期望的梯度
        print("\n  Computing expected gradients manually...")
        grad_output = torch.ones_like(y)
        print(f"    grad_output shape: {grad_output.shape}")
        detailed_check(grad_output, "grad_output")
        
        # weight.grad应该等于 x.t() @ grad_output，然后转置
        x_flat_for_grad = x.view(-1, x.size(-1))  # (20, 768)
        grad_output_flat = grad_output.view(-1, grad_output.size(-1))  # (20, 3072)
        print(f"    x_flat_for_grad shape: {x_flat_for_grad.shape}")
        print(f"    grad_output_flat shape: {grad_output_flat.shape}")
        
        try:
            expected_weight_grad = torch.mm(x_flat_for_grad.t(), grad_output_flat)  # (768, 3072)
            print(f"    expected_weight_grad shape: {expected_weight_grad.shape}")
            expected_grad_has_issue = detailed_check(expected_weight_grad, "expected weight grad")
            if expected_grad_has_issue:
                print("    ✗ Expected gradient computation produces NaN/Inf!")
                print("    This suggests the problem is in mm operation during gradient computation")
        except Exception as e:
            print(f"    ✗ Expected gradient computation failed: {e}")
        
        loss.backward()
        
        print(f"  Weight grad after backward: {linear.weight.grad is not None}")
        print(f"  Bias grad after backward: {linear.bias.grad is not None}")
        
        if linear.weight.grad is not None:
            print(f"  Weight grad device: {linear.weight.grad.device}")
            print(f"  Weight grad shape: {linear.weight.grad.shape}")
            
            # 检查梯度tensor的内存是否被正确初始化
            print("\n  Checking gradient tensor initialization...")
            grad_cpu = linear.weight.grad.cpu()
            zero_count = (grad_cpu == 0).sum().item()
            nan_count = torch.isnan(grad_cpu).sum().item()
            inf_count = torch.isinf(grad_cpu).sum().item()
            print(f"    Zero values: {zero_count}/{grad_cpu.numel()} ({zero_count/grad_cpu.numel()*100:.2f}%)")
            print(f"    NaN values: {nan_count}/{grad_cpu.numel()} ({nan_count/grad_cpu.numel()*100:.2f}%)")
            print(f"    Inf values: {inf_count}/{grad_cpu.numel()} ({inf_count/grad_cpu.numel()*100:.2f}%)")
            
            # 检查NaN的分布模式
            if nan_count > 0:
                nan_mask = torch.isnan(grad_cpu)
                # 检查NaN是否集中在某些行或列
                nan_rows = nan_mask.any(dim=1).sum().item()
                nan_cols = nan_mask.any(dim=0).sum().item()
                print(f"    NaN rows: {nan_rows}/{grad_cpu.shape[0]}")
                print(f"    NaN cols: {nan_cols}/{grad_cpu.shape[1]}")
            
            grad_has_issue = detailed_check(linear.weight.grad, "Weight gradient")
            if grad_has_issue:
                print("\n✗ Weight gradient has NaN/Inf!")
                return False
        else:
            print("  ⚠ Weight gradient is None")
        
        if linear.bias.grad is not None:
            print(f"  Bias grad device: {linear.bias.grad.device}")
            print(f"  Bias grad shape: {linear.bias.grad.shape}")
            grad_has_issue = detailed_check(linear.bias.grad, "Bias gradient")
            if grad_has_issue:
                print("\n✗ Bias gradient has NaN/Inf!")
                return False
        else:
            print("  ⚠ Bias gradient is None")
        
        print("  ✓ Backward pass OK")
    except Exception as e:
        print(f"\n✗ Backward pass failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓ All steps passed!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_linear_step_by_step()
    sys.exit(0 if success else 1)

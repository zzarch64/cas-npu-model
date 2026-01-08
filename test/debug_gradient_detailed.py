#!/usr/bin/env python
"""
详细调试梯度计算过程
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

def check_tensor(tensor, name):
    """检查tensor的详细信息"""
    if tensor is None:
        print(f"  {name}: None")
        return
    cpu_t = tensor.cpu() if tensor.device.type == 'cas_npu' else tensor
    has_nan = torch.isnan(cpu_t).any().item()
    has_inf = torch.isinf(cpu_t).any().item()
    status = "✗ NaN/Inf" if (has_nan or has_inf) else "✓ OK"
    print(f"  {name}: {status}, device={tensor.device}, shape={tensor.shape}")
    if has_nan:
        nan_count = torch.isnan(cpu_t).sum().item()
        print(f"    NaN: {nan_count}/{cpu_t.numel()} ({nan_count/cpu_t.numel()*100:.2f}%)")
    if has_inf:
        inf_count = torch.isinf(cpu_t).sum().item()
        print(f"    Inf: {inf_count}/{cpu_t.numel()} ({inf_count/cpu_t.numel()*100:.2f}%)")
    if not has_nan and not has_inf:
        print(f"    min={cpu_t.min().item():.6f}, max={cpu_t.max().item():.6f}, mean={cpu_t.mean().item():.6f}")

def test_gradient_flow():
    """测试梯度流动过程"""
    print("=" * 80)
    print("Detailed Gradient Flow Debug")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    # 创建简单的Linear层
    linear = nn.Linear(10, 5).to(device)
    x = torch.randn(2, 10, device=device, requires_grad=True)
    
    print("\n[Initial State]")
    check_tensor(linear.weight, "weight")
    check_tensor(linear.bias, "bias")
    check_tensor(x, "input")
    
    # 前向传播
    print("\n[Forward Pass]")
    y = linear(x)
    check_tensor(y, "output")
    
    # 计算损失
    loss = y.sum()
    print(f"\n[Loss]")
    print(f"  loss: {loss.item():.6f}")
    
    # 手动计算期望的梯度
    print("\n[Expected Gradients]")
    grad_output = torch.ones_like(y)  # loss = y.sum(), so grad_output = ones
    check_tensor(grad_output, "grad_output (should be ones)")
    
    # weight.grad应该等于 x.t() @ grad_output
    # 注意：Linear层使用weight.t()，所以实际梯度需要转置
    expected_weight_grad = torch.mm(x.t(), grad_output)  # shape: [10, 5]
    check_tensor(expected_weight_grad, "expected weight.grad (before transpose)")
    print(f"    Note: Linear uses weight.t(), so actual grad will be transposed")
    
    # bias.grad应该等于 grad_output.sum(0)
    expected_bias_grad = grad_output.sum(0)
    check_tensor(expected_bias_grad, "expected bias.grad")
    
    # 执行反向传播
    print("\n[Backward Pass]")
    print("  Executing loss.backward()...")
    
    # 在反向传播前，检查中间tensor
    print("\n  Before backward:")
    print(f"    y.grad_fn: {y.grad_fn}")
    if y.grad_fn:
        print(f"      type: {type(y.grad_fn).__name__}")
        print(f"      next_functions: {y.grad_fn.next_functions}")
    
    loss.backward()
    
    print("\n  After backward:")
    check_tensor(linear.weight.grad, "weight.grad")
    check_tensor(linear.bias.grad, "bias.grad")
    check_tensor(x.grad, "input.grad")
    
    # 比较实际梯度和期望梯度
    if linear.weight.grad is not None:
        print("\n[Gradient Comparison]")
        actual_cpu = linear.weight.grad.cpu()
        # Linear层使用weight.t()，所以梯度需要转置
        expected_cpu = expected_weight_grad.t().cpu()
        
        print(f"  Actual weight.grad shape: {actual_cpu.shape}")
        print(f"  Expected weight.grad shape: {expected_cpu.shape}")
        
        # 检查shape是否匹配
        if actual_cpu.shape != expected_cpu.shape:
            print(f"  ⚠ Shape mismatch! Cannot compare directly.")
            print(f"    This might be due to Linear layer implementation differences.")
        else:
            # 检查NaN的位置
            if torch.isnan(actual_cpu).any():
                nan_mask = torch.isnan(actual_cpu)
                nan_indices = torch.nonzero(nan_mask, as_tuple=False)
                print(f"  NaN positions (first 10):")
                for i, idx in enumerate(nan_indices[:10]):
                    if len(idx) == 2:
                        print(f"    [{idx[0].item()}, {idx[1].item()}]")
                        # 检查对应的期望值
                        if idx[0].item() < expected_cpu.shape[0] and idx[1].item() < expected_cpu.shape[1]:
                            exp_val = expected_cpu[idx[0].item(), idx[1].item()]
                            print(f"      Expected value: {exp_val.item():.6f}")
            
            # 检查非NaN部分是否匹配
            valid_mask = ~torch.isnan(actual_cpu) & ~torch.isinf(actual_cpu)
            if valid_mask.any():
                actual_valid = actual_cpu[valid_mask]
                expected_valid = expected_cpu[valid_mask]
                if len(actual_valid) == len(expected_valid):
                    diff = (actual_valid - expected_valid).abs()
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    print(f"  Valid values comparison:")
                    print(f"    Valid elements: {len(actual_valid)}/{actual_cpu.numel()}")
                    print(f"    Max diff: {max_diff:.6f}")
                    print(f"    Mean diff: {mean_diff:.6f}")
                    if max_diff < 1e-4:  # 放宽阈值，因为可能有数值误差
                        print(f"    ✓ Valid gradients match!")
                    else:
                        print(f"    ⚠ Valid gradients have some difference (might be numerical error)")
                else:
                    print(f"  ⚠ Cannot compare: valid mask sizes don't match")
    
    # 检查梯度计算过程中使用的操作
    print("\n[Gradient Computation Operations]")
    print("  Checking if mm operations are used in gradient computation...")
    
    # 手动模拟梯度计算过程
    print("\n  Manual gradient computation simulation:")
    
    # 1. weight.grad = x.t() @ grad_output
    print("    Step 1: Computing weight.grad = x.t() @ grad_output")
    x_t = x.t()
    check_tensor(x_t, "x.t()")
    check_tensor(grad_output, "grad_output")
    
    try:
        manual_weight_grad = torch.mm(x_t, grad_output)
        check_tensor(manual_weight_grad, "manual weight.grad (x.t() @ grad_output)")
        
        # 检查这个结果是否包含NaN
        if torch.isnan(manual_weight_grad).any():
            print("    ✗ Manual computation produces NaN!")
            print("    This suggests mm operation has a problem in backward context")
        else:
            print("    ✓ Manual computation OK")
    except Exception as e:
        print(f"    ✗ Manual computation failed: {e}")
    
    # 2. bias.grad = grad_output.sum(0)
    print("\n    Step 2: Computing bias.grad = grad_output.sum(0)")
    try:
        manual_bias_grad = grad_output.sum(0)
        check_tensor(manual_bias_grad, "manual bias.grad (grad_output.sum(0))")
    except Exception as e:
        print(f"    ✗ Manual computation failed: {e}")

if __name__ == "__main__":
    test_gradient_flow()

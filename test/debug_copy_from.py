#!/usr/bin/env python
"""
测试_copy_from操作，特别是大tensor的拷贝
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

def test_copy_from_large_tensor():
    """测试大tensor的拷贝"""
    print("=" * 80)
    print("_copy_from Large Tensor Test")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    # 创建大tensor（模拟梯度tensor的大小）
    size = (3072, 768)
    cpu_tensor = torch.randn(*size)
    
    print(f"\n[Creating CPU tensor]")
    print(f"  Shape: {cpu_tensor.shape}")
    print(f"  Device: {cpu_tensor.device}")
    
    # 检查CPU tensor
    has_nan = torch.isnan(cpu_tensor).any().item()
    has_inf = torch.isinf(cpu_tensor).any().item()
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    
    # 拷贝到设备
    print(f"\n[Copying to device]")
    print(f"  Executing: device_tensor = cpu_tensor.to(device)...")
    
    try:
        device_tensor = cpu_tensor.to(device)
        
        print(f"  ✓ Copy completed")
        print(f"  Device tensor shape: {device_tensor.shape}")
        print(f"  Device tensor device: {device_tensor.device}")
        print(f"  Is contiguous: {device_tensor.is_contiguous()}")
        
        # 检查设备tensor
        device_cpu = device_tensor.cpu()
        has_nan = torch.isnan(device_cpu).any().item()
        has_inf = torch.isinf(device_cpu).any().item()
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        
        if has_nan or has_inf:
            print("  ✗ Device tensor contains NaN/Inf!")
            return False
        
        # 验证数据是否正确
        diff = (cpu_tensor - device_cpu).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-5:
            print("  ✓ Data matches!")
        else:
            print("  ✗ Data doesn't match!")
            return False
        
        # 测试反向拷贝（Device -> CPU）
        print(f"\n[Copying back to CPU]")
        device_tensor2 = torch.randn(*size, device=device)
        cpu_tensor2 = device_tensor2.cpu()
        
        device_cpu2 = device_tensor2.cpu()
        has_nan2 = torch.isnan(device_cpu2).any().item()
        has_inf2 = torch.isinf(device_cpu2).any().item()
        print(f"  Has NaN: {has_nan2}")
        print(f"  Has Inf: {has_inf2}")
        
        if has_nan2 or has_inf2:
            print("  ✗ CPU tensor contains NaN/Inf!")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Copy failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_copy_from_partial():
    """测试部分拷贝（模拟梯度计算过程中的情况）"""
    print("\n" + "=" * 80)
    print("_copy_from Partial Copy Test")
    print("=" * 80)
    
    device = torch.device('cas_npu:0')
    
    # 创建设备tensor（模拟梯度tensor）
    device_tensor = torch.zeros(3072, 768, device=device)
    
    print(f"\n[Initial device tensor]")
    print(f"  Shape: {device_tensor.shape}")
    print(f"  Is contiguous: {device_tensor.is_contiguous()}")
    
    # 创建CPU tensor（模拟梯度计算结果）
    cpu_tensor = torch.randn(3072, 768)
    
    # 只更新部分行（模拟梯度计算过程中某些行没有被计算）
    cpu_tensor[100:200, :] = float('nan')  # 模拟某些行包含NaN
    
    print(f"\n[CPU tensor with NaN]")
    nan_count = torch.isnan(cpu_tensor).sum().item()
    print(f"  NaN count: {nan_count}/{cpu_tensor.numel()}")
    
    # 拷贝到设备
    print(f"\n[Copying to device]")
    device_tensor.copy_(cpu_tensor)
    
    # 检查设备tensor
    device_cpu = device_tensor.cpu()
    nan_count_after = torch.isnan(device_cpu).sum().item()
    print(f"  NaN count after copy: {nan_count_after}/{device_cpu.numel()}")
    
    if nan_count_after == nan_count:
        print("  ✓ NaN correctly copied")
    else:
        print(f"  ✗ NaN count mismatch: expected {nan_count}, got {nan_count_after}")
        return False
    
    return True

if __name__ == "__main__":
    success1 = test_copy_from_large_tensor()
    success2 = test_copy_from_partial()
    
    print("\n" + "=" * 80)
    if success1 and success2:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 80)
    
    sys.exit(0 if (success1 and success2) else 1)

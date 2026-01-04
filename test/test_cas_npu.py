#!/usr/bin/env python
"""
CAS-NPU自定义设备测试脚本

测试内容：
1. 设备可用性检查
2. Tensor创建和设备转移
3. add.Tensor操作
4. 设备切换
5. Tensor方法

注意：请在PyTorch源码目录外运行此测试！
"""

import sys
import os

# 确保导入本地的cas_npu包（从test目录向上一级找到cas_npu包）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入cas_npu扩展 - 这会自动完成后端注册
import cas_npu

import torch
print(f"PyTorch version: {torch.__version__}")


def test_device_availability():
    """测试1: 设备可用性"""
    print("\n" + "=" * 50)
    print("Test 1: Device Availability")
    print("=" * 50)
    
    # 检查设备是否可用
    available = torch.cas_npu.is_available()
    print(f"CAS-NPU available: {available}")
    assert available, "CAS-NPU device should be available"
    
    # 获取设备数量
    count = torch.cas_npu.device_count()
    print(f"Device count: {count}")
    assert count == 1, f"Expected 1 device, got {count}"
    
    print("✓ Device availability test passed")


def test_tensor_creation():
    """测试2: Tensor创建和设备转移"""
    print("\n" + "=" * 50)
    print("Test 2: Tensor Creation and Transfer")
    print("=" * 50)
    
    # 在CPU上创建tensor
    cpu_tensor = torch.randn(3, 4, dtype=torch.float32)
    print(f"CPU tensor: shape={cpu_tensor.shape}, device={cpu_tensor.device}")
    
    # 转移到CAS-NPU设备
    device = torch.device("cas_npu:0")
    npu_tensor = cpu_tensor.to(device)
    print(f"CAS-NPU tensor: shape={npu_tensor.shape}, device={npu_tensor.device}")
    
    # 验证设备类型
    assert npu_tensor.device.type == "cas_npu", f"Expected cas_npu device, got {npu_tensor.device.type}"
    assert npu_tensor.device.index == 0, f"Expected device index 0, got {npu_tensor.device.index}"
    
    # 转回CPU
    cpu_tensor_back = npu_tensor.cpu()
    print(f"Back to CPU: shape={cpu_tensor_back.shape}, device={cpu_tensor_back.device}")
    
    # 验证数据一致性
    assert torch.allclose(cpu_tensor, cpu_tensor_back), "Data should be the same after round-trip"
    
    print("✓ Tensor creation test passed")


def test_add_tensor():
    """测试3: add.Tensor操作"""
    print("\n" + "=" * 50)
    print("Test 3: add.Tensor Operation")
    print("=" * 50)
    
    device = torch.device("cas_npu:0")
    
    # 创建测试数据
    a_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b_cpu = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    
    print(f"a_cpu:\n{a_cpu}")
    print(f"b_cpu:\n{b_cpu}")
    
    # 转到CAS-NPU设备
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    
    print(f"\na on {a.device}")
    print(f"b on {b.device}")
    
    # 执行加法
    c = a + b
    print(f"\nResult c = a + b on {c.device}:")
    
    # 验证结果
    c_cpu = c.cpu()
    expected = a_cpu + b_cpu
    print(f"c_cpu:\n{c_cpu}")
    print(f"expected:\n{expected}")
    
    assert torch.allclose(c_cpu, expected), "Add operation result mismatch"
    
    # 测试带alpha的加法
    d = torch.add(a, b, alpha=2.0)
    d_cpu = d.cpu()
    expected_alpha = a_cpu + 2.0 * b_cpu
    print(f"\nResult d = a + 2*b:")
    print(f"d_cpu:\n{d_cpu}")
    print(f"expected:\n{expected_alpha}")
    
    assert torch.allclose(d_cpu, expected_alpha), "Add with alpha operation result mismatch"
    
    print("✓ add.Tensor test passed")


def test_device_switching():
    """测试4: 设备切换"""
    print("\n" + "=" * 50)
    print("Test 4: Device Switching")
    print("=" * 50)
    
    # 获取当前设备
    current = torch.cas_npu.current_device()
    print(f"Current device: {current}")
    assert current == 0, f"Expected device 0, got {current}"
    
    # 在设备0上创建tensor
    device0 = torch.device("cas_npu:0")
    t0_cpu = torch.randn(2, 2)
    t0 = t0_cpu.to(device0)
    print(f"Tensor on device 0: {t0.device}")
    assert t0.device.index == 0, f"Expected device index 0, got {t0.device.index}"
    
    print("✓ Device switching test passed")


def test_tensor_methods():
    """测试5: Tensor方法（is_cas_npu等）"""
    print("\n" + "=" * 50)
    print("Test 5: Tensor Methods")
    print("=" * 50)
    
    cpu_tensor = torch.randn(2, 3)
    npu_tensor = cpu_tensor.to("cas_npu:0")
    
    # 测试is_cas_npu属性
    print(f"cpu_tensor.is_cas_npu: {cpu_tensor.is_cas_npu}")
    print(f"npu_tensor.is_cas_npu: {npu_tensor.is_cas_npu}")
    
    assert not cpu_tensor.is_cas_npu, "CPU tensor should not be cas_npu"
    assert npu_tensor.is_cas_npu, "NPU tensor should be cas_npu"
    
    # 测试cas_npu()方法
    t = torch.randn(2, 2)
    t_npu = t.cas_npu()
    print(f"t.cas_npu() device: {t_npu.device}")
    assert t_npu.is_cas_npu, "Tensor after .cas_npu() should be on cas_npu device"
    
    print("✓ Tensor methods test passed")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("CAS-NPU Custom Device Extension Tests")
    print("=" * 60)
    
    try:
        test_device_availability()
        test_tensor_creation()
        test_add_tensor()
        test_device_switching()
        test_tensor_methods()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

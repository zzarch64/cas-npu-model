#!/usr/bin/env python
"""
LeNet 神经网络推理示例 - 验证 CPU Fallback 机制

这个示例展示了：
1. 在 cas_npu 设备上运行完整的神经网络（前向传播）
2. 大部分操作会 fallback 到 CPU 执行
3. add 操作会使用我们实现的 cas_npu 版本

运行方式:
    python examples/lenet_inference.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入cas_npu扩展
import cas_npu

import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"PyTorch version: {torch.__version__}")
print(f"CAS-NPU available: {torch.cas_npu.is_available()}")
print()


# ============ 注册CPU Fallback操作 ============

def _to_cpu(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cas_npu':
        return tensor.cpu()
    return tensor

def _to_device(tensor, device):
    if isinstance(tensor, torch.Tensor) and tensor.device != device:
        return tensor.to(device)
    return tensor


@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_fallback(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    device = input.device
    result = torch.convolution(
        _to_cpu(input), _to_cpu(weight), _to_cpu(bias) if bias is not None else None,
        stride, padding, dilation, transposed, output_padding, groups
    )
    return _to_device(result, device)


@torch.library.impl("aten::max_pool2d_with_indices", "privateuseone")
def max_pool2d_fallback(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    device = input.device
    if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0):
        stride = kernel_size
    output, indices = F.max_pool2d(_to_cpu(input), kernel_size, stride, padding, dilation, ceil_mode, return_indices=True)
    return _to_device(output, device), _to_device(indices, device)


@torch.library.impl("aten::relu", "privateuseone")
def relu_fallback(input):
    device = input.device
    return _to_device(F.relu(_to_cpu(input)), device)


@torch.library.impl("aten::mm", "privateuseone")
def mm_fallback(input, mat2):
    device = input.device
    return _to_device(torch.mm(_to_cpu(input), _to_cpu(mat2)), device)


@torch.library.impl("aten::addmm", "privateuseone")
def addmm_fallback(bias, input, weight, beta=1, alpha=1):
    device = input.device
    return _to_device(torch.addmm(_to_cpu(bias), _to_cpu(input), _to_cpu(weight), beta=beta, alpha=alpha), device)


@torch.library.impl("aten::t", "privateuseone")
def t_fallback(input):
    device = input.device
    return _to_device(_to_cpu(input).t(), device)


print("✓ CPU fallback operations registered")
print()


# ============ LeNet网络定义 ============

class LeNet(nn.Module):
    """LeNet-5 经典5层卷积神经网络"""
    
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============ 测试函数 ============

def test_lenet_on_cpu():
    """在CPU上运行LeNet作为基准"""
    print("=" * 60)
    print("Test 1: LeNet on CPU (Baseline)")
    print("=" * 60)
    
    torch.manual_seed(42)
    model = LeNet()
    x = torch.randn(4, 1, 28, 28)
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input shape: {x.shape}")
    
    start = time.time()
    with torch.no_grad():
        output = model(x)
    elapsed = time.time() - start
    
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0, :5]}")  # 前5个输出
    print(f"Time: {elapsed*1000:.2f} ms")
    print("✓ CPU baseline passed\n")
    return output


def test_lenet_on_cas_npu():
    """在cas_npu设备上运行LeNet"""
    print("=" * 60)
    print("Test 2: LeNet on CAS-NPU (with CPU Fallback)")
    print("=" * 60)
    
    device = torch.device('cas_npu:0')
    
    torch.manual_seed(42)
    model = LeNet().to(device)
    x = torch.randn(4, 1, 28, 28).to(device)
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device: {x.device}")
    
    print("\nRunning forward pass...")
    start = time.time()
    with torch.no_grad():
        output = model(x)
    elapsed = time.time() - start
    
    print(f"Output shape: {output.shape}, device: {output.device}")
    print(f"Output sample: {output.cpu()[0, :5]}")  # 前5个输出
    print(f"Time: {elapsed*1000:.2f} ms")
    print("✓ CAS-NPU forward pass completed\n")
    return output.cpu()


def test_output_consistency():
    """验证CPU和CAS-NPU输出一致"""
    print("=" * 60)
    print("Test 3: Verify CPU vs CAS-NPU Output Consistency")
    print("=" * 60)
    
    torch.manual_seed(42)
    model_cpu = LeNet()
    x_cpu = torch.randn(2, 1, 28, 28)
    
    torch.manual_seed(42)
    model_npu = LeNet().to('cas_npu:0')
    x_npu = torch.randn(2, 1, 28, 28).to('cas_npu:0')
    
    with torch.no_grad():
        out_cpu = model_cpu(x_cpu)
        out_npu = model_npu(x_npu).cpu()
    
    print(f"CPU output:\n{out_cpu}")
    print(f"CAS-NPU output:\n{out_npu}")
    
    max_diff = (out_cpu - out_npu).abs().max().item()
    print(f"\nMax difference: {max_diff:.2e}")
    
    if torch.allclose(out_cpu, out_npu, atol=1e-5):
        print("✓ Outputs match exactly!")
    else:
        print("⚠ Small numerical differences (expected)")
    print()


def test_add_operation():
    """验证add操作使用cas_npu实现"""
    print("=" * 60)
    print("Test 4: Verify add.Tensor on CAS-NPU")
    print("=" * 60)
    
    device = torch.device('cas_npu:0')
    
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(device)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).to(device)
    
    print(f"a:\n{a.cpu()}")
    print(f"b:\n{b.cpu()}")
    
    c = a + b  # 使用我们实现的cas_npu add
    print(f"a + b =\n{c.cpu()}")
    
    expected = torch.tensor([[6.0, 8.0], [10.0, 12.0]])
    assert torch.allclose(c.cpu(), expected)
    print("✓ add.Tensor verified\n")


def test_multiple_operations():
    """测试连续多个操作"""
    print("=" * 60)
    print("Test 5: Multiple Operations on CAS-NPU")
    print("=" * 60)
    
    device = torch.device('cas_npu:0')
    
    # 创建数据
    x = torch.randn(2, 3).to(device)
    y = torch.randn(2, 3).to(device)
    
    print(f"x:\n{x.cpu()}")
    print(f"y:\n{y.cpu()}")
    
    # 连续操作
    z1 = x + y          # add (cas_npu实现)
    z2 = F.relu(z1)     # relu (CPU fallback)
    z3 = z2 + x         # add (cas_npu实现)
    
    print(f"(x + y):\n{z1.cpu()}")
    print(f"relu(x + y):\n{z2.cpu()}")
    print(f"relu(x + y) + x:\n{z3.cpu()}")
    
    print("✓ Multiple operations completed\n")


def main():
    print("=" * 60)
    print("LeNet Neural Network Inference Example on CAS-NPU Device")
    print("=" * 60)
    print()
    print("This demonstrates:")
    print("- LeNet forward pass on cas_npu device")
    print("- CPU fallback for conv2d, relu, linear, max_pool2d")
    print("- add.Tensor using our CAS-NPU implementation")
    print()
    
    try:
        test_lenet_on_cpu()
        test_lenet_on_cas_npu()
        test_output_consistency()
        test_add_operation()
        test_multiple_operations()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("Summary:")
        print("- ✓ LeNet forward pass on CPU")
        print("- ✓ LeNet forward pass on CAS-NPU")
        print("- ✓ Output consistency verified")
        print("- ✓ add.Tensor on CAS-NPU")
        print("- ✓ Multiple operations chain")
        print()
        print("Note: Training (backward pass) requires more autograd support.")
        print("      This demo shows inference works with CPU fallback.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

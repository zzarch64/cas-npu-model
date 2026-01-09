#!/usr/bin/env python
"""
LeNet 训练示例 - 使用 MSELoss 避免 log_softmax 问题

这个示例展示了：
1. 在 cas_npu 设备上运行完整的训练流程
2. 反向传播和梯度计算
3. 参数更新和优化器使用

由于 AutogradPrivateUse1 需要在 C++ 中更完整地注册，
这里使用 MSELoss 来演示训练流程（避免 CrossEntropyLoss 中的 log_softmax）

运行方式:
    python examples/lenet_training.py
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


# ============ 工具函数 ============

def _to_cpu(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cas_npu':
        return tensor.cpu()
    return tensor

def _to_device(tensor, device):
    if isinstance(tensor, torch.Tensor) and tensor.device != device:
        return tensor.to(device)
    return tensor


# ============ 注册核心操作 ============

@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_fallback(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    device = input.device
    result = torch.convolution(
        _to_cpu(input), _to_cpu(weight), _to_cpu(bias) if bias is not None else None,
        stride, padding, dilation, transposed, output_padding, groups
    )
    return _to_device(result, device)


@torch.library.impl("aten::convolution_backward_overrideable", "privateuseone")
def convolution_backward_fallback(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask):
    device = grad_output.device
    grad_input = grad_weight = grad_bias = None
    
    if output_mask[0]:
        grad_input = torch.nn.grad.conv2d_input(
            _to_cpu(input).shape, _to_cpu(weight), _to_cpu(grad_output), stride, padding, dilation, groups
        )
        grad_input = _to_device(grad_input, device)
    
    if output_mask[1]:
        grad_weight = torch.nn.grad.conv2d_weight(
            _to_cpu(input), _to_cpu(weight).shape, _to_cpu(grad_output), stride, padding, dilation, groups
        )
        grad_weight = _to_device(grad_weight, device)
    
    if output_mask[2] and _to_cpu(grad_output).dim() == 4:
        grad_bias = _to_cpu(grad_output).sum(dim=[0, 2, 3])
        grad_bias = _to_device(grad_bias, device)
    
    return grad_input, grad_weight, grad_bias


@torch.library.impl("aten::max_pool2d_with_indices", "privateuseone")
def max_pool2d_fallback(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    device = input.device
    if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0):
        stride = kernel_size
    output, indices = F.max_pool2d(_to_cpu(input), kernel_size, stride, padding, dilation, ceil_mode, return_indices=True)
    return _to_device(output, device), _to_device(indices, device)


@torch.library.impl("aten::max_pool2d_with_indices_backward", "privateuseone")
def max_pool2d_backward_fallback(grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices):
    device = grad_output.device
    if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0):
        stride = kernel_size
    result = F.max_unpool2d(_to_cpu(grad_output), _to_cpu(indices), kernel_size, stride, padding, _to_cpu(input).shape[-2:])
    return _to_device(result, device)


@torch.library.impl("aten::relu", "privateuseone")
def relu_fallback(input):
    device = input.device
    return _to_device(F.relu(_to_cpu(input)), device)


@torch.library.impl("aten::threshold_backward", "privateuseone")
def threshold_backward_fallback(grad_output, input, threshold):
    device = grad_output.device
    result = _to_cpu(grad_output) * (_to_cpu(input) > threshold).float()
    return _to_device(result, device)


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


@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided_fallback(input, size, stride, storage_offset=None):
    device = input.device
    result = torch.as_strided(_to_cpu(input), size, stride, storage_offset)
    return _to_device(result, device)


@torch.library.impl("aten::expand", "privateuseone")
def expand_fallback(input, size, implicit=False):
    device = input.device
    result = _to_cpu(input).expand(size)
    return _to_device(result, device)


@torch.library.impl("aten::sum", "privateuseone")
def sum_fallback(input, dtype=None):
    device = input.device
    result = _to_cpu(input).sum(dtype=dtype)
    return _to_device(result, device)


@torch.library.impl("aten::sum.dim_IntList", "privateuseone")
def sum_dim_fallback(input, dim, keepdim=False, dtype=None):
    device = input.device
    result = _to_cpu(input).sum(dim=dim, keepdim=keepdim, dtype=dtype)
    return _to_device(result, device)


@torch.library.impl("aten::mul.Tensor", "privateuseone")
def mul_tensor_fallback(input, other):
    device = input.device
    result = _to_cpu(input) * _to_cpu(other)
    return _to_device(result, device)


@torch.library.impl("aten::mul.Scalar", "privateuseone")
def mul_scalar_fallback(input, other):
    device = input.device
    result = _to_cpu(input) * other
    return _to_device(result, device)


@torch.library.impl("aten::div.Tensor", "privateuseone")
def div_tensor_fallback(input, other):
    device = input.device
    result = _to_cpu(input) / _to_cpu(other)
    return _to_device(result, device)


@torch.library.impl("aten::sub.Tensor", "privateuseone")
def sub_tensor_fallback(input, other, alpha=1):
    device = input.device
    result = _to_cpu(input) - alpha * _to_cpu(other)
    return _to_device(result, device)


@torch.library.impl("aten::neg", "privateuseone")
def neg_fallback(input):
    device = input.device
    result = -_to_cpu(input)
    return _to_device(result, device)


@torch.library.impl("aten::pow.Tensor_Scalar", "privateuseone")
def pow_fallback(input, exponent):
    device = input.device
    result = _to_cpu(input).pow(exponent)
    return _to_device(result, device)


@torch.library.impl("aten::mean", "privateuseone")
def mean_fallback(input, dtype=None):
    device = input.device
    result = _to_cpu(input).mean(dtype=dtype)
    return _to_device(result, device)


@torch.library.impl("aten::mean.dim", "privateuseone")
def mean_dim_fallback(input, dim, keepdim=False, dtype=None):
    device = input.device
    result = _to_cpu(input).mean(dim=dim, keepdim=keepdim, dtype=dtype)
    return _to_device(result, device)


print("✓ CPU fallback operations registered")
print()


# ============ LeNet网络 ============

class LeNet(nn.Module):
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

def test_add_backward():
    """测试add操作的反向传播"""
    print("=" * 60)
    print("Test 1: add.Tensor Backward")
    print("=" * 60)
    
    device = torch.device('cas_npu:0')
    
    # 在CPU创建带梯度的tensor，然后转到设备
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    
    a_dev = a.to(device)
    b_dev = b.to(device)
    
    c = a_dev + b_dev
    loss = c.sum()
    loss.backward()
    
    print(f"a + b = {c.cpu().detach()}")
    print(f"a.grad = {a.grad}")
    print(f"b.grad = {b.grad}")
    
    # 验证梯度
    assert torch.allclose(a.grad, torch.ones(2, 2)), "a gradient incorrect"
    assert torch.allclose(b.grad, torch.ones(2, 2)), "b gradient incorrect"
    
    print("✓ add.Tensor backward verified\n")


def test_linear_backward():
    """测试线性层的反向传播"""
    print("=" * 60)
    print("Test 2: Linear Layer Backward")
    print("=" * 60)
    
    device = torch.device('cas_npu:0')
    
    # 简单的线性层
    linear = nn.Linear(4, 2).to(device)
    x = torch.randn(2, 4).to(device)
    
    # 前向
    y = linear(x)
    loss = y.pow(2).sum()  # 使用MSE类型的loss
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Loss: {loss.cpu().item():.4f}")
    
    # 反向
    loss.backward()
    
    # 检查梯度
    print(f"Weight grad shape: {linear.weight.grad.shape}")
    print(f"Bias grad shape: {linear.bias.grad.shape}")
    print(f"Weight grad norm: {linear.weight.grad.cpu().norm().item():.4f}")
    
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None
    
    print("✓ Linear backward verified\n")


def test_conv_backward():
    """测试卷积层的反向传播"""
    print("=" * 60)
    print("Test 3: Conv2d Layer Backward")
    print("=" * 60)
    
    device = torch.device('cas_npu:0')
    
    conv = nn.Conv2d(1, 2, kernel_size=3, padding=1).to(device)
    x = torch.randn(1, 1, 4, 4).to(device)
    
    y = conv(x)
    loss = y.pow(2).sum()
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Loss: {loss.cpu().item():.4f}")
    
    loss.backward()
    
    print(f"Weight grad shape: {conv.weight.grad.shape}")
    print(f"Weight grad norm: {conv.weight.grad.cpu().norm().item():.4f}")
    
    assert conv.weight.grad is not None
    
    print("✓ Conv2d backward verified\n")


def test_lenet_training():
    """测试LeNet完整训练（使用MSELoss）"""
    print("=" * 60)
    print("Test 4: LeNet Training with MSELoss")
    print("=" * 60)
    
    device = torch.device('cas_npu:0')
    
    model = LeNet().to(device)
    criterion = nn.MSELoss()  # 使用MSELoss避免log_softmax问题
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 假数据 - 使用one-hot编码的target
    x = torch.randn(4, 1, 28, 28).to(device)
    target_idx = torch.randint(0, 10, (4,))
    target = torch.zeros(4, 10).scatter_(1, target_idx.unsqueeze(1), 1.0).to(device)
    
    print(f"Training on {device}")
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {target.shape}")
    
    losses = []
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.cpu().item()
        losses.append(loss_val)
        print(f"Epoch {epoch+1}, Loss: {loss_val:.4f}")
    
    # 验证loss在下降
    assert losses[-1] < losses[0], "Loss should decrease during training"
    
    print("✓ LeNet training completed, loss decreased\n")


def test_gradient_accumulation():
    """测试梯度累积"""
    print("=" * 60)
    print("Test 5: Gradient Accumulation")
    print("=" * 60)
    
    device = torch.device('cas_npu:0')
    
    linear = nn.Linear(4, 2).to(device)
    
    # 第一次前向反向
    x1 = torch.randn(2, 4).to(device)
    y1 = linear(x1)
    loss1 = y1.sum()
    loss1.backward()
    
    grad1 = linear.weight.grad.clone()
    
    # 第二次前向反向（不清零梯度）
    x2 = torch.randn(2, 4).to(device)
    y2 = linear(x2)
    loss2 = y2.sum()
    loss2.backward()
    
    grad2 = linear.weight.grad
    
    print(f"Grad after 1st backward: norm = {grad1.cpu().norm().item():.4f}")
    print(f"Grad after 2nd backward: norm = {grad2.cpu().norm().item():.4f}")
    
    # 梯度应该累积
    assert grad2.cpu().norm() > grad1.cpu().norm() * 0.5, "Gradients should accumulate"
    
    print("✓ Gradient accumulation verified\n")


def main():
    print("=" * 60)
    print("LeNet Training Test on CAS-NPU Device")
    print("=" * 60)
    print()
    print("Note: Using MSELoss instead of CrossEntropyLoss")
    print("      (CrossEntropyLoss requires more Autograd registrations)")
    print()
    
    try:
        test_add_backward()
        test_linear_backward()
        test_conv_backward()
        test_lenet_training()
        test_gradient_accumulation()
        
        print("=" * 60)
        print("All training tests passed! ✓")
        print("=" * 60)
        print()
        print("Summary:")
        print("- ✓ add.Tensor backward")
        print("- ✓ Linear layer backward")
        print("- ✓ Conv2d layer backward")
        print("- ✓ LeNet training (MSELoss)")
        print("- ✓ Gradient accumulation")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

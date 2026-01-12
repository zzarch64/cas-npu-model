#!/usr/bin/env python
"""
详细测试 addmm 操作

检查 addmm 在 FFN 中的使用是否正确
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cas_npu

device_npu = torch.device('cas_npu:0')

print("=" * 60)
print("Testing addmm operation")
print("=" * 60)

# addmm: output = beta * input + alpha * (mat1 @ mat2)
# 在 PyTorch 中，linear layer 使用 addmm: output = input @ weight^T + bias

# 测试 1: 基本 addmm
print("\n1. Basic addmm test...")
batch_size = 1
seq_len = 3
hidden_size = 896
out_features = 3584

input_cpu = torch.randn(batch_size * seq_len, hidden_size, dtype=torch.float32)
weight_cpu = torch.randn(out_features, hidden_size, dtype=torch.float32)
bias_cpu = torch.randn(out_features, dtype=torch.float32)

input_npu = input_cpu.to(device_npu)
weight_npu = weight_cpu.to(device_npu)
bias_npu = bias_cpu.to(device_npu)

# addmm(input, weight^T, bias) = input @ weight^T + bias
result_cpu = torch.addmm(bias_cpu, input_cpu, weight_cpu.t())
result_npu = torch.addmm(bias_npu, input_npu, weight_npu.t()).cpu()

diff = (result_cpu - result_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"  ✗ WARNING: addmm differs!")
    print(f"  CPU result (first 5): {result_cpu[0, :5]}")
    print(f"  NPU result (first 5): {result_npu[0, :5]}")
    print(f"  Difference (first 5): {diff[0, :5]}")
else:
    print(f"  ✓ addmm matches")

# 测试 2: 使用实际的模型权重
print("\n2. Testing with actual model weights...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

model_cpu = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float32,
    local_files_only=True,
)
model_cpu.eval()

model_npu = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float32,
    local_files_only=True,
)
model_npu = model_npu.to(device_npu)
model_npu.eval()

# 获取第一层 FFN 的权重
first_layer = model_cpu.model.layers[0]
ffn_gate_proj = first_layer.mlp.gate_proj
ffn_up_proj = first_layer.mlp.up_proj
ffn_down_proj = first_layer.mlp.down_proj

# 创建测试输入
test_input = torch.randn(1, 3, 896, dtype=torch.float32)
test_input_2d = test_input.view(-1, 896)

print("\n  Testing gate_proj...")
gate_weight_cpu = ffn_gate_proj.weight.data  # [intermediate_size, hidden_size]
gate_bias_cpu = ffn_gate_proj.bias.data if ffn_gate_proj.bias is not None else None

gate_weight_npu = gate_weight_cpu.to(device_npu)
gate_bias_npu = gate_bias_cpu.to(device_npu) if gate_bias_cpu is not None else None

# Linear: output = input @ weight^T + bias
gate_out_cpu = torch.addmm(
    gate_bias_cpu if gate_bias_cpu is not None else torch.zeros(gate_weight_cpu.size(0)),
    test_input_2d,
    gate_weight_cpu.t()
)
gate_out_npu = torch.addmm(
    gate_bias_npu if gate_bias_npu is not None else torch.zeros(gate_weight_npu.size(0), device=device_npu),
    test_input_2d.to(device_npu),
    gate_weight_npu.t()
).cpu()

diff = (gate_out_cpu - gate_out_npu).abs()
print(f"    Max difference: {diff.max().item():.6f}")
print(f"    Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"    ✗ gate_proj differs!")
    print(f"    Max diff at: {diff.argmax()}")
    idx = diff.argmax()
    row = idx // diff.shape[1]
    col = idx % diff.shape[1]
    print(f"    CPU value: {gate_out_cpu[row, col].item():.6f}")
    print(f"    NPU value: {gate_out_npu[row, col].item():.6f}")
else:
    print(f"    ✓ gate_proj matches")

print("\n  Testing up_proj...")
up_weight_cpu = ffn_up_proj.weight.data
up_bias_cpu = ffn_up_proj.bias.data if ffn_up_proj.bias is not None else None

up_weight_npu = up_weight_cpu.to(device_npu)
up_bias_npu = up_bias_cpu.to(device_npu) if up_bias_cpu is not None else None

up_out_cpu = torch.addmm(
    up_bias_cpu if up_bias_cpu is not None else torch.zeros(up_weight_cpu.size(0)),
    test_input_2d,
    up_weight_cpu.t()
)
up_out_npu = torch.addmm(
    up_bias_npu if up_bias_npu is not None else torch.zeros(up_weight_npu.size(0), device=device_npu),
    test_input_2d.to(device_npu),
    up_weight_npu.t()
).cpu()

diff = (up_out_cpu - up_out_npu).abs()
print(f"    Max difference: {diff.max().item():.6f}")
print(f"    Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"    ✗ up_proj differs!")
else:
    print(f"    ✓ up_proj matches")

print("\n  Testing down_proj...")
down_weight_cpu = ffn_down_proj.weight.data
down_bias_cpu = ffn_down_proj.bias.data if ffn_down_proj.bias is not None else None

down_weight_npu = down_weight_cpu.to(device_npu)
down_bias_npu = down_bias_cpu.to(device_npu) if down_bias_cpu is not None else None

# 先计算 gate 和 up 的输出（模拟 FFN 的前半部分）
gate_out_2d = gate_out_cpu
up_out_2d = up_out_cpu
# SiLU(gate) * up
silu_gate = gate_out_2d * torch.sigmoid(gate_out_2d)
ffn_intermediate = silu_gate * up_out_2d

# down_proj
down_out_cpu = torch.addmm(
    down_bias_cpu if down_bias_cpu is not None else torch.zeros(down_weight_cpu.size(0)),
    ffn_intermediate,
    down_weight_cpu.t()
)

# NPU 版本
gate_out_2d_npu = gate_out_npu.to(device_npu)
up_out_2d_npu = up_out_npu.to(device_npu)
silu_gate_npu = gate_out_2d_npu * torch.sigmoid(gate_out_2d_npu)
ffn_intermediate_npu = silu_gate_npu * up_out_2d_npu

down_out_npu = torch.addmm(
    down_bias_npu if down_bias_npu is not None else torch.zeros(down_weight_npu.size(0), device=device_npu),
    ffn_intermediate_npu,
    down_weight_npu.t()
).cpu()

diff = (down_out_cpu - down_out_npu).abs()
print(f"    Max difference: {diff.max().item():.6f}")
print(f"    Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"    ✗ down_proj differs!")
    print(f"    Max diff at: {diff.argmax()}")
    idx = diff.argmax()
    row = idx // diff.shape[1]
    col = idx % diff.shape[1]
    print(f"    CPU value: {down_out_cpu[row, col].item():.6f}")
    print(f"    NPU value: {down_out_npu[row, col].item():.6f}")
else:
    print(f"    ✓ down_proj matches")

print("\n✓ All tests completed!")

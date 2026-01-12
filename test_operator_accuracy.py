#!/usr/bin/env python
"""
系统化测试各个算子的准确性

逐步测试每个关键算子，找出哪个算子导致输出差异
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cas_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

device_npu = torch.device('cas_npu:0')
device_cpu = torch.device('cpu')

model_path = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("=" * 60)
print("Test 1: 基础算子测试 (mm, bmm, add)")
print("=" * 60)

# 测试 mm (矩阵乘法)
print("\n1. Testing mm (matrix multiplication)...")
A_cpu = torch.randn(3, 4, dtype=torch.float32)
B_cpu = torch.randn(4, 5, dtype=torch.float32)
A_npu = A_cpu.to(device_npu)
B_npu = B_cpu.to(device_npu)

result_cpu = torch.mm(A_cpu, B_cpu)
result_npu = torch.mm(A_npu, B_npu).cpu()

diff = (result_cpu - result_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"  ✗ WARNING: mm output differs!")
    print(f"  CPU result (first 3x3):\n{result_cpu[:3, :3]}")
    print(f"  NPU result (first 3x3):\n{result_npu[:3, :3]}")
else:
    print(f"  ✓ mm output matches")

# 测试 bmm (批量矩阵乘法)
print("\n2. Testing bmm (batch matrix multiplication)...")
A_cpu = torch.randn(2, 3, 4, dtype=torch.float32)
B_cpu = torch.randn(2, 4, 5, dtype=torch.float32)
A_npu = A_cpu.to(device_npu)
B_npu = B_cpu.to(device_npu)

result_cpu = torch.bmm(A_cpu, B_cpu)
result_npu = torch.bmm(A_npu, B_npu).cpu()

diff = (result_cpu - result_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"  ✗ WARNING: bmm output differs!")
    print(f"  CPU result (first batch, first 2x2):\n{result_cpu[0, :2, :2]}")
    print(f"  NPU result (first batch, first 2x2):\n{result_npu[0, :2, :2]}")
else:
    print(f"  ✓ bmm output matches")

# 测试 add
print("\n3. Testing add...")
A_cpu = torch.randn(3, 4, dtype=torch.float32)
B_cpu = torch.randn(3, 4, dtype=torch.float32)
A_npu = A_cpu.to(device_npu)
B_npu = B_cpu.to(device_npu)

result_cpu = torch.add(A_cpu, B_cpu)
result_npu = torch.add(A_npu, B_npu).cpu()

diff = (result_cpu - result_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"  ✗ WARNING: add output differs!")
else:
    print(f"  ✓ add output matches")

# 测试 addmm
print("\n4. Testing addmm...")
A_cpu = torch.randn(3, 5, dtype=torch.float32)
B_cpu = torch.randn(3, 4, dtype=torch.float32)
C_cpu = torch.randn(4, 5, dtype=torch.float32)
A_npu = A_cpu.to(device_npu)
B_npu = B_cpu.to(device_npu)
C_npu = C_cpu.to(device_npu)

result_cpu = torch.addmm(A_cpu, B_cpu, C_cpu)
result_npu = torch.addmm(A_npu, B_npu, C_npu).cpu()

diff = (result_cpu - result_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"  ✗ WARNING: addmm output differs!")
else:
    print(f"  ✓ addmm output matches")

print("\n" + "=" * 60)
print("Test 2: 模型第一层输出对比")
print("=" * 60)

# 加载模型
print("\nLoading models...")
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

# 准备输入
prompt = "讲个笑话"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", None)

print(f"\nPrompt: {prompt}")
print(f"Input IDs: {input_ids.tolist()}")

# Hook 第一层的输出
print("\nHooking first transformer layer...")

# 获取第一个 transformer layer
first_layer_cpu = model_cpu.model.layers[0]
first_layer_npu = model_npu.model.layers[0]

# Hook forward
outputs_cpu_layer = []
outputs_npu_layer = []

def hook_cpu(module, input, output):
    outputs_cpu_layer.append(output[0].detach().clone())

def hook_npu(module, input, output):
    outputs_npu_layer.append(output[0].detach().clone())

handle_cpu = first_layer_cpu.register_forward_hook(hook_cpu)
handle_npu = first_layer_npu.register_forward_hook(hook_npu)

# Forward pass
print("Forward pass on CPU...")
with torch.no_grad():
    outputs_cpu = model_cpu(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

print("Forward pass on CAS-NPU...")
input_ids_npu = input_ids.to(device_npu)
attention_mask_npu = attention_mask.to(device_npu) if attention_mask is not None else None
with torch.no_grad():
    outputs_npu = model_npu(
        input_ids=input_ids_npu,
        attention_mask=attention_mask_npu,
    )

# 比较第一层输出
if outputs_cpu_layer and outputs_npu_layer:
    layer_out_cpu = outputs_cpu_layer[0]
    layer_out_npu = outputs_npu_layer[0].cpu()
    
    diff = (layer_out_cpu - layer_out_npu).abs()
    print(f"\nFirst layer output comparison:")
    print(f"  CPU shape: {layer_out_cpu.shape}")
    print(f"  NPU shape: {layer_out_npu.shape}")
    print(f"  Max difference: {diff.max().item():.6f}")
    print(f"  Mean difference: {diff.mean().item():.6f}")
    
    if diff.max().item() > 0.1:
        print(f"  ✗ WARNING: First layer output differs significantly!")
        if layer_out_cpu.dim() == 3:
            print(f"  CPU output (first 5 values): {layer_out_cpu[0, 0, :5]}")
            print(f"  NPU output (first 5 values): {layer_out_npu[0, 0, :5]}")
        elif layer_out_cpu.dim() == 2:
            print(f"  CPU output (first 5 values): {layer_out_cpu[0, :5]}")
            print(f"  NPU output (first 5 values): {layer_out_npu[0, :5]}")
    else:
        print(f"  ✓ First layer output matches")

# 比较最终输出
logits_cpu = outputs_cpu.logits[0, -1, :10]
logits_npu = outputs_npu.logits[0, -1, :10].cpu()

diff = (logits_cpu - logits_npu).abs()
print(f"\nFinal logits comparison:")
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
print(f"  CPU logits (first 10): {logits_cpu}")
print(f"  NPU logits (first 10): {logits_npu}")

handle_cpu.remove()
handle_npu.remove()

print("\n" + "=" * 60)
print("Test 3: 逐步检查每个 transformer layer")
print("=" * 60)

# 逐步检查每一层
print("\nChecking each transformer layer...")
num_layers = len(model_cpu.model.layers)

layer_diffs = []
for i in range(min(3, num_layers)):  # 只检查前3层
    layer_cpu = model_cpu.model.layers[i]
    layer_npu = model_npu.model.layers[i]
    
    outputs_cpu_layer = []
    outputs_npu_layer = []
    
    def hook_cpu(module, input, output):
        outputs_cpu_layer.append(output[0].detach().clone())
    
    def hook_npu(module, input, output):
        outputs_npu_layer.append(output[0].detach().clone())
    
    handle_cpu = layer_cpu.register_forward_hook(hook_cpu)
    handle_npu = layer_npu.register_forward_hook(hook_npu)
    
    # Forward pass
    with torch.no_grad():
        if i == 0:
            hidden_cpu = model_cpu.model.embed_tokens(input_ids)
            hidden_npu = model_npu.model.embed_tokens(input_ids_npu)
        else:
            hidden_cpu = layer_diffs[i-1]['hidden_cpu']
            hidden_npu = layer_diffs[i-1]['hidden_npu']
        
        _ = layer_cpu(hidden_cpu, attention_mask=attention_mask)
        _ = layer_npu(hidden_npu, attention_mask=attention_mask_npu)
    
    if outputs_cpu_layer and outputs_npu_layer:
        out_cpu = outputs_cpu_layer[0]
        out_npu = outputs_npu_layer[0].cpu()
        
        diff = (out_cpu - out_npu).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        layer_diffs.append({
            'layer': i,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'hidden_cpu': out_cpu,
            'hidden_npu': out_npu.to(device_npu),
        })
        
        print(f"  Layer {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        if max_diff > 0.1:
            print(f"    ✗ WARNING: Layer {i} output differs significantly!")
    
    handle_cpu.remove()
    handle_npu.remove()

print("\n✓ All tests completed!")

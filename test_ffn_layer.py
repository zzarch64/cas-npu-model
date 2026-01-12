#!/usr/bin/env python
"""
检查 FFN (Feed Forward Network) 层的计算

定位问题是否在 FFN 层
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cas_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

device_npu = torch.device('cas_npu:0')

model_path = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading models...")
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

prompt = "讲个笑话"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", None)

input_ids_npu = input_ids.to(device_npu)
attention_mask_npu = attention_mask.to(device_npu) if attention_mask is not None else None

print(f"\nPrompt: {prompt}")
print(f"Input IDs: {input_ids.tolist()}")

print("\n" + "=" * 60)
print("Testing first transformer layer components")
print("=" * 60)

# Hook 第一层的各个组件
first_layer_cpu = model_cpu.model.layers[0]
first_layer_npu = model_npu.model.layers[0]

# 存储各部分的输出
attn_out_cpu = None
attn_out_npu = None
ffn_out_cpu = None
ffn_out_npu = None
layer_out_cpu = None
layer_out_npu = None

def hook_attn_cpu(module, input, output):
    global attn_out_cpu
    attn_out_cpu = output[0].detach().clone()

def hook_attn_npu(module, input, output):
    global attn_out_npu
    attn_out_npu = output[0].detach().clone()

def hook_ffn_cpu(module, input, output):
    global ffn_out_cpu
    ffn_out_cpu = output.detach().clone()

def hook_ffn_npu(module, input, output):
    global ffn_out_npu
    ffn_out_npu = output.detach().clone()

def hook_layer_cpu(module, input, output):
    global layer_out_cpu
    layer_out_cpu = output[0].detach().clone()

def hook_layer_npu(module, input, output):
    global layer_out_npu
    layer_out_npu = output[0].detach().clone()

# 注册 hooks
handle_attn_cpu = first_layer_cpu.self_attn.register_forward_hook(hook_attn_cpu)
handle_attn_npu = first_layer_npu.self_attn.register_forward_hook(hook_attn_npu)
handle_ffn_cpu = first_layer_cpu.mlp.register_forward_hook(hook_ffn_cpu)
handle_ffn_npu = first_layer_npu.mlp.register_forward_hook(hook_ffn_npu)
handle_layer_cpu = first_layer_cpu.register_forward_hook(hook_layer_cpu)
handle_layer_npu = first_layer_npu.register_forward_hook(hook_layer_npu)

# Forward pass
print("\nForward pass...")
with torch.no_grad():
    outputs_cpu = model_cpu(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    outputs_npu = model_npu(
        input_ids=input_ids_npu,
        attention_mask=attention_mask_npu,
    )

# 比较各部分输出
print("\n1. Attention output:")
if attn_out_cpu is not None and attn_out_npu is not None:
    diff = (attn_out_cpu - attn_out_npu.cpu()).abs()
    print(f"  Max difference: {diff.max().item():.6f}")
    print(f"  Mean difference: {diff.mean().item():.6f}")
    if diff.max().item() > 0.1:
        print(f"  ✗ Attention output differs significantly!")
    else:
        print(f"  ✓ Attention output matches")

print("\n2. FFN output:")
if ffn_out_cpu is not None and ffn_out_npu is not None:
    diff = (ffn_out_cpu - ffn_out_npu.cpu()).abs()
    print(f"  Max difference: {diff.max().item():.6f}")
    print(f"  Mean difference: {diff.mean().item():.6f}")
    if diff.max().item() > 0.1:
        print(f"  ✗ WARNING: FFN output differs significantly!")
        # 找出差异最大的位置
        max_idx = diff.argmax()
        if ffn_out_cpu.dim() == 3:
            idx_0, idx_1, idx_2 = torch.unravel_index(max_idx, diff.shape)
            print(f"  Max diff at [{idx_0}, {idx_1}, {idx_2}]")
            print(f"  CPU value: {ffn_out_cpu[idx_0, idx_1, idx_2].item():.6f}")
            print(f"  NPU value: {ffn_out_npu[idx_0, idx_1, idx_2].cpu().item():.6f}")
    else:
        print(f"  ✓ FFN output matches")

print("\n3. Layer output (after residual and layer norm):")
if layer_out_cpu is not None and layer_out_npu is not None:
    diff = (layer_out_cpu - layer_out_npu.cpu()).abs()
    print(f"  Max difference: {diff.max().item():.6f}")
    print(f"  Mean difference: {diff.mean().item():.6f}")
    if diff.max().item() > 0.1:
        print(f"  ✗ WARNING: Layer output differs significantly!")
    else:
        print(f"  ✓ Layer output matches")

# 比较最终 logits
print("\n4. Final logits:")
logits_cpu = outputs_cpu.logits[0, -1, :10]
logits_npu = outputs_npu.logits[0, -1, :10].cpu()

diff = (logits_cpu - logits_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
print(f"  CPU logits (first 10): {logits_cpu}")
print(f"  NPU logits (first 10): {logits_npu}")

# 清理
handle_attn_cpu.remove()
handle_attn_npu.remove()
handle_ffn_cpu.remove()
handle_ffn_npu.remove()
handle_layer_cpu.remove()
handle_layer_npu.remove()

# 测试 FFN 中的关键操作
print("\n" + "=" * 60)
print("Testing FFN operations")
print("=" * 60)

# FFN 通常包含：linear1 -> activation -> linear2
print("\n1. Testing linear layers (mm)...")
# 模拟 FFN 的第一层
batch_size = 1
seq_len = 3
hidden_size = 896
intermediate_size = 3584  # 通常是 hidden_size * 4

# 创建权重（模拟 linear layer）
weight_cpu = torch.randn(intermediate_size, hidden_size, dtype=torch.float32)
bias_cpu = torch.randn(intermediate_size, dtype=torch.float32)
input_cpu = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

weight_npu = weight_cpu.to(device_npu)
bias_npu = bias_cpu.to(device_npu)
input_npu = input_cpu.to(device_npu)

# Linear: output = input @ weight^T + bias
# 需要 reshape 为 (batch*seq, hidden) @ (hidden, intermediate)
input_2d_cpu = input_cpu.view(-1, hidden_size)
input_2d_npu = input_npu.view(-1, hidden_size)

output_cpu = torch.mm(input_2d_cpu, weight_cpu.t()) + bias_cpu
output_npu = torch.mm(input_2d_npu, weight_npu.t()) + bias_npu

diff = (output_cpu - output_npu.cpu()).abs()
print(f"  Max difference: {diff.max().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"  ✗ Linear layer differs!")
else:
    print(f"  ✓ Linear layer matches")

print("\n2. Testing activation (SiLU/Swish)...")
# SiLU: x * sigmoid(x)
x_cpu = torch.randn(10, dtype=torch.float32)
x_npu = x_cpu.to(device_npu)

silu_cpu = x_cpu * torch.sigmoid(x_cpu)
silu_npu = x_npu * torch.sigmoid(x_npu)

diff = (silu_cpu - silu_npu.cpu()).abs()
print(f"  Max difference: {diff.max().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"  ✗ Activation differs!")
else:
    print(f"  ✓ Activation matches")

print("\n✓ All tests completed!")

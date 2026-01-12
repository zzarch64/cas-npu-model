#!/usr/bin/env python
"""
逐步检查 FFN 的每个计算步骤

定位具体哪个步骤导致差异
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

# 获取第一层的输入（attention 输出）
print("\nGetting attention output...")
first_layer_cpu = model_cpu.model.layers[0]
first_layer_npu = model_npu.model.layers[0]

attn_out_cpu = None
attn_out_npu = None

def hook_attn_cpu(module, input, output):
    global attn_out_cpu
    attn_out_cpu = output[0].detach().clone()

def hook_attn_npu(module, input, output):
    global attn_out_npu
    attn_out_npu = output[0].detach().clone()

handle_attn_cpu = first_layer_cpu.self_attn.register_forward_hook(hook_attn_cpu)
handle_attn_npu = first_layer_npu.self_attn.register_forward_hook(hook_attn_npu)

with torch.no_grad():
    _ = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
    _ = model_npu(input_ids=input_ids_npu, attention_mask=attention_mask_npu)

handle_attn_cpu.remove()
handle_attn_npu.remove()

print(f"  Attention output shape: {attn_out_cpu.shape}")
attn_diff = (attn_out_cpu - attn_out_npu.cpu()).abs()
print(f"  Attention output max diff: {attn_diff.max().item():.6f}")

# 现在逐步检查 FFN
print("\n" + "=" * 60)
print("Step-by-step FFN check")
print("=" * 60)

# FFN 输入（attention 输出 + residual）
# 在 Qwen2 中，FFN 输入是 attention 输出经过 layer norm 后的结果
input_norm_cpu = first_layer_cpu.input_layernorm(attn_out_cpu)
input_norm_npu = first_layer_npu.input_layernorm(attn_out_npu)

print("\n1. Input layer norm:")
diff = (input_norm_cpu - input_norm_npu.cpu()).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 0.1:
    print(f"  ✗ Input layer norm differs!")
else:
    print(f"  ✓ Input layer norm matches")

# FFN: gate_proj, up_proj, SiLU, multiply, down_proj
ffn_cpu = first_layer_cpu.mlp
ffn_npu = first_layer_npu.mlp

print("\n2. Gate projection:")
gate_out_cpu = ffn_cpu.gate_proj(input_norm_cpu)
gate_out_npu = ffn_npu.gate_proj(input_norm_npu).cpu()

diff = (gate_out_cpu - gate_out_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 0.1:
    print(f"  ✗ Gate projection differs!")
    max_idx = diff.argmax()
    if gate_out_cpu.dim() == 3:
        idx_0, idx_1, idx_2 = torch.unravel_index(max_idx, diff.shape)
        print(f"  Max diff at [{idx_0}, {idx_1}, {idx_2}]")
        print(f"  CPU: {gate_out_cpu[idx_0, idx_1, idx_2].item():.6f}")
        print(f"  NPU: {gate_out_npu[idx_0, idx_1, idx_2].item():.6f}")
else:
    print(f"  ✓ Gate projection matches")

print("\n3. Up projection:")
up_out_cpu = ffn_cpu.up_proj(input_norm_cpu)
up_out_npu = ffn_npu.up_proj(input_norm_npu).cpu()

diff = (up_out_cpu - up_out_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 0.1:
    print(f"  ✗ Up projection differs!")
else:
    print(f"  ✓ Up projection matches")

print("\n4. SiLU activation (on gate):")
silu_gate_cpu = torch.nn.functional.silu(gate_out_cpu)
silu_gate_npu = torch.nn.functional.silu(gate_out_npu.to(device_npu)).cpu()

diff = (silu_gate_cpu - silu_gate_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 0.1:
    print(f"  ✗ SiLU differs!")
else:
    print(f"  ✓ SiLU matches")

print("\n5. Multiply (SiLU(gate) * up):")
multiply_cpu = silu_gate_cpu * up_out_cpu
multiply_npu = silu_gate_npu * up_out_npu

diff = (multiply_cpu - multiply_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 0.1:
    print(f"  ✗ Multiply differs!")
else:
    print(f"  ✓ Multiply matches")

print("\n6. Down projection:")
down_out_cpu = ffn_cpu.down_proj(multiply_cpu)
down_out_npu = ffn_npu.down_proj(multiply_npu.to(device_npu)).cpu()

diff = (down_out_cpu - down_out_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 0.1:
    print(f"  ✗ Down projection differs!")
    max_idx = diff.argmax()
    if down_out_cpu.dim() == 3:
        idx_0, idx_1, idx_2 = torch.unravel_index(max_idx, diff.shape)
        print(f"  Max diff at [{idx_0}, {idx_1}, {idx_2}]")
        print(f"  CPU: {down_out_cpu[idx_0, idx_1, idx_2].item():.6f}")
        print(f"  NPU: {down_out_npu[idx_0, idx_1, idx_2].item():.6f}")
else:
    print(f"  ✓ Down projection matches")

# 检查完整的 FFN 输出
print("\n7. Complete FFN output:")
ffn_out_cpu = ffn_cpu(input_norm_cpu)
ffn_out_npu = ffn_npu(input_norm_npu).cpu()

diff = (ffn_out_cpu - ffn_out_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 0.1:
    print(f"  ✗ Complete FFN output differs!")
else:
    print(f"  ✓ Complete FFN output matches")

print("\n✓ Analysis completed!")

#!/usr/bin/env python
"""
专门测试 attention 计算部分

检查 attention 计算中使用的算子是否正确
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
print("Testing attention computation")
print("=" * 60)

# Hook attention 层的输入和输出
attention_inputs_cpu = []
attention_outputs_cpu = []
attention_inputs_npu = []
attention_outputs_npu = []

def hook_attention_cpu(module, input, output):
    attention_inputs_cpu.append([x.detach().clone() if isinstance(x, torch.Tensor) else x for x in input])
    attention_outputs_cpu.append(output[0].detach().clone())

def hook_attention_npu(module, input, output):
    attention_inputs_npu.append([x.detach().clone() if isinstance(x, torch.Tensor) else x for x in input])
    attention_outputs_npu.append(output[0].detach().clone())

# 注册 hook 到第一层的 attention
first_layer_cpu = model_cpu.model.layers[0]
first_layer_npu = model_npu.model.layers[0]

handle_cpu = first_layer_cpu.self_attn.register_forward_hook(hook_attention_cpu)
handle_npu = first_layer_npu.self_attn.register_forward_hook(hook_attention_npu)

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

# 比较 attention 输入
if attention_inputs_cpu and attention_inputs_npu:
    print("\n1. Attention input comparison:")
    for i, (in_cpu, in_npu) in enumerate(zip(attention_inputs_cpu[0], attention_inputs_npu[0])):
        if isinstance(in_cpu, torch.Tensor) and isinstance(in_npu, torch.Tensor):
            in_npu_cpu = in_npu.cpu()
            diff = (in_cpu - in_npu_cpu).abs()
            print(f"  Input {i}:")
            print(f"    CPU shape: {in_cpu.shape}")
            print(f"    NPU shape: {in_npu_cpu.shape}")
            print(f"    Max difference: {diff.max().item():.6f}")
            if diff.max().item() > 1e-5:
                print(f"    ✗ Input differs!")
            else:
                print(f"    ✓ Input matches")

# 比较 attention 输出
if attention_outputs_cpu and attention_outputs_npu:
    print("\n2. Attention output comparison:")
    out_cpu = attention_outputs_cpu[0]
    out_npu = attention_outputs_npu[0].cpu()
    
    diff = (out_cpu - out_npu).abs()
    print(f"  CPU shape: {out_cpu.shape}")
    print(f"  NPU shape: {out_npu.shape}")
    print(f"  Max difference: {diff.max().item():.6f}")
    print(f"  Mean difference: {diff.mean().item():.6f}")
    
    if diff.max().item() > 0.1:
        print(f"  ✗ WARNING: Attention output differs significantly!")
        # 找出差异最大的位置
        max_idx = diff.argmax()
        if out_cpu.dim() == 3:
            idx_0, idx_1, idx_2 = torch.unravel_index(max_idx, diff.shape)
            print(f"  Max diff at [{idx_0}, {idx_1}, {idx_2}]")
            print(f"  CPU value: {out_cpu[idx_0, idx_1, idx_2].item():.6f}")
            print(f"  NPU value: {out_npu[idx_0, idx_1, idx_2].item():.6f}")
    else:
        print(f"  ✓ Attention output matches")

# 比较最终 logits
print("\n3. Final logits comparison:")
logits_cpu = outputs_cpu.logits[0, -1, :10]
logits_npu = outputs_npu.logits[0, -1, :10].cpu()

diff = (logits_cpu - logits_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
print(f"  CPU logits (first 10): {logits_cpu}")
print(f"  NPU logits (first 10): {logits_npu}")

handle_cpu.remove()
handle_npu.remove()

# 测试 attention 中的关键操作
print("\n" + "=" * 60)
print("Testing attention operations")
print("=" * 60)

# 模拟 attention 计算
print("\n1. Testing Q @ K^T (bmm)...")
batch_size = 1
seq_len = 3
head_dim = 64
num_heads = 8

Q_cpu = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
K_cpu = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
Q_npu = Q_cpu.to(device_npu)
K_npu = K_cpu.to(device_npu)

# Q @ K^T
QK_cpu = torch.bmm(Q_cpu.view(batch_size * num_heads, seq_len, head_dim),
                   K_cpu.view(batch_size * num_heads, seq_len, head_dim).transpose(1, 2))
QK_npu = torch.bmm(Q_npu.view(batch_size * num_heads, seq_len, head_dim),
                   K_npu.view(batch_size * num_heads, seq_len, head_dim).transpose(1, 2))

diff = (QK_cpu - QK_npu.cpu()).abs()
print(f"  Max difference: {diff.max().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"  ✗ QK^T differs!")
else:
    print(f"  ✓ QK^T matches")

# 测试 softmax
print("\n2. Testing softmax...")
scores_cpu = torch.randn(batch_size * num_heads, seq_len, seq_len, dtype=torch.float32)
scores_npu = scores_cpu.to(device_npu)

softmax_cpu = torch.softmax(scores_cpu, dim=-1)
softmax_npu = torch.softmax(scores_npu, dim=-1).cpu()

diff = (softmax_cpu - softmax_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"  ✗ Softmax differs!")
else:
    print(f"  ✓ Softmax matches")

# 测试 attention @ V
print("\n3. Testing attention @ V (bmm)...")
attn_cpu = softmax_cpu
V_cpu = torch.randn(batch_size * num_heads, seq_len, head_dim, dtype=torch.float32)
V_npu = V_cpu.to(device_npu)

attnV_cpu = torch.bmm(attn_cpu, V_cpu)
attnV_npu = torch.bmm(softmax_npu.to(device_npu), V_npu).cpu()

diff = (attnV_cpu - attnV_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"  ✗ Attention @ V differs!")
else:
    print(f"  ✓ Attention @ V matches")

print("\n✓ All tests completed!")

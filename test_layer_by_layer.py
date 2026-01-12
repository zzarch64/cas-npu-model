#!/usr/bin/env python
"""
逐层检查模型，定位问题出现的具体位置
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
print("Step-by-step comparison")
print("=" * 60)

# 1. Embedding
print("\n1. Embedding layer...")
with torch.no_grad():
    emb_cpu = model_cpu.model.embed_tokens(input_ids)
    emb_npu = model_npu.model.embed_tokens(input_ids_npu).cpu()

diff = (emb_cpu - emb_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
if diff.max().item() > 1e-5:
    print(f"  ✗ Embedding differs!")
else:
    print(f"  ✓ Embedding matches")

# 2. 逐层检查
hidden_cpu = emb_cpu
hidden_npu = emb_npu.to(device_npu)

num_layers = len(model_cpu.model.layers)
print(f"\n2. Checking {num_layers} transformer layers...")

for i in range(min(5, num_layers)):  # 检查前5层
    layer_cpu = model_cpu.model.layers[i]
    layer_npu = model_npu.model.layers[i]
    
    print(f"\n  Layer {i}:")
    
    # Forward pass
    with torch.no_grad():
        out_cpu = layer_cpu(hidden_cpu, attention_mask=attention_mask)[0]
        out_npu = layer_npu(hidden_npu, attention_mask=attention_mask_npu)[0]
    
    diff = (out_cpu - out_npu.cpu()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"    Max difference: {max_diff:.6f}")
    print(f"    Mean difference: {mean_diff:.6f}")
    
    if max_diff > 0.1:
        print(f"    ✗ WARNING: Significant difference detected!")
        # 找出差异最大的位置
        max_idx = diff.argmax()
        if out_cpu.dim() == 3:
            idx_0, idx_1, idx_2 = torch.unravel_index(max_idx, diff.shape)
            print(f"    Max diff at [{idx_0}, {idx_1}, {idx_2}]")
            print(f"    CPU value: {out_cpu[idx_0, idx_1, idx_2].item():.6f}")
            print(f"    NPU value: {out_npu[idx_0, idx_1, idx_2].cpu().item():.6f}")
    else:
        print(f"    ✓ Output matches")
    
    hidden_cpu = out_cpu
    hidden_npu = out_npu

# 3. 最终输出
print("\n3. Final output...")
with torch.no_grad():
    # 使用已经处理过的 hidden states
    # 但我们需要完整的 forward pass
    outputs_cpu = model_cpu(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    outputs_npu = model_npu(
        input_ids=input_ids_npu,
        attention_mask=attention_mask_npu,
    )

logits_cpu = outputs_cpu.logits[0, -1, :10]
logits_npu = outputs_npu.logits[0, -1, :10].cpu()

diff = (logits_cpu - logits_npu).abs()
print(f"  Max difference: {diff.max().item():.6f}")
print(f"  Mean difference: {diff.mean().item():.6f}")
print(f"  CPU logits (first 10): {logits_cpu}")
print(f"  NPU logits (first 10): {logits_npu}")

print("\n✓ Analysis completed!")

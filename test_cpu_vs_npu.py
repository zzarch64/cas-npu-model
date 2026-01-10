#!/usr/bin/env python
"""
比较 CPU 和 CAS-NPU 上的模型输出

检查模型在两种设备上的输出是否一致
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cas_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "讲个笑话"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", None)

print("=" * 60)
print("Test: CPU vs CAS-NPU forward pass comparison")
print("=" * 60)

# CPU 模型
print("\nLoading CPU model...")
model_cpu = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float32,
    local_files_only=True,
)
model_cpu.eval()

# CAS-NPU 模型
print("Loading CAS-NPU model...")
model_npu = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float32,
    local_files_only=True,
)
model_npu = model_npu.to('cas_npu:0')
model_npu.eval()

print(f"\nPrompt: {prompt}")
print(f"Input IDs: {input_ids.tolist()}")

# Forward pass on CPU
print("\nForward pass on CPU...")
with torch.no_grad():
    outputs_cpu = model_cpu(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits_cpu = outputs_cpu.logits[0, -1, :10]  # 最后一个token的前10个logits
    print(f"Last token logits (first 10): {logits_cpu}")

# Forward pass on CAS-NPU
print("\nForward pass on CAS-NPU...")
input_ids_npu = input_ids.to('cas_npu:0')
attention_mask_npu = attention_mask.to('cas_npu:0') if attention_mask is not None else None
with torch.no_grad():
    outputs_npu = model_npu(
        input_ids=input_ids_npu,
        attention_mask=attention_mask_npu,
    )
    logits_npu = outputs_npu.logits[0, -1, :10].cpu()  # 最后一个token的前10个logits
    print(f"Last token logits (first 10): {logits_npu}")

# 比较
print("\nComparison:")
diff = (logits_cpu - logits_npu).abs()
print(f"Absolute difference: {diff}")
print(f"Max difference: {diff.max().item()}")
print(f"Mean difference: {diff.mean().item()}")

if diff.max().item() < 0.01:
    print("✓ Outputs are very similar (diff < 0.01)")
elif diff.max().item() < 1.0:
    print("⚠ Outputs differ but within reasonable range (diff < 1.0)")
else:
    print("✗ WARNING: Outputs differ significantly (diff >= 1.0)")

# 检查生成的 token
print("\n" + "=" * 60)
print("Test: Generation comparison")
print("=" * 60)

print("\nGenerating on CPU...")
with torch.no_grad():
    outputs_cpu = model_cpu.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        do_sample=False,
    )
text_cpu = tokenizer.decode(outputs_cpu[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"Generated: {text_cpu}")

print("\nGenerating on CAS-NPU...")
with torch.no_grad():
    outputs_npu = model_npu.generate(
        input_ids=input_ids_npu,
        attention_mask=attention_mask_npu,
        max_new_tokens=5,
        do_sample=False,
    )
text_npu = tokenizer.decode(outputs_npu[0][input_ids_npu.shape[1]:], skip_special_tokens=True)
print(f"Generated: {text_npu}")

if text_cpu == text_npu:
    print("\n✓ Generated texts match!")
else:
    print(f"\n✗ Generated texts differ!")
    print(f"  CPU: {text_cpu}")
    print(f"  NPU: {text_npu}")

print("\n✓ All tests completed!")

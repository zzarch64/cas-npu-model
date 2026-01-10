#!/usr/bin/env python
"""
详细测试 attention_mask 在模型内部的使用

检查：
1. attention_mask 是否被转换为 attention bias
2. masked_fill_ 是否被正确调用
3. 模型输出是否受到 attention_mask 的影响
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cas_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cas_npu:0')

# 加载模型
print("Loading model...")
model_path = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float32,
    local_files_only=True,
)
model = model.to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n" + "=" * 60)
print("Test: 检查 attention_mask 对输出的影响")
print("=" * 60)

# 创建两个相同的输入，但不同的 attention_mask
prompt = "讲个笑话"
inputs = tokenizer(prompt, return_tensors="pt", padding=False)
input_ids = inputs["input_ids"].to(device)

# 测试 1: 完整的 attention_mask（所有位置都是1）
attention_mask_full = torch.ones_like(input_ids).to(device)

# 测试 2: 只关注前一半的 attention_mask
seq_len = input_ids.shape[1]
attention_mask_half = torch.zeros_like(input_ids).to(device)
attention_mask_half[0, :seq_len//2] = 1

print(f"Prompt: {prompt}")
print(f"Input IDs: {input_ids.cpu().tolist()}")
print(f"Full attention mask: {attention_mask_full.cpu().tolist()}")
print(f"Half attention mask: {attention_mask_half.cpu().tolist()}")

# 执行 forward pass
print("\nForward pass with full attention_mask:")
with torch.no_grad():
    outputs_full = model(
        input_ids=input_ids,
        attention_mask=attention_mask_full,
    )
    logits_full = outputs_full.logits[0, -1, :10]  # 最后一个token的前10个logits
    print(f"Last token logits (first 10): {logits_full.cpu()}")

print("\nForward pass with half attention_mask:")
with torch.no_grad():
    outputs_half = model(
        input_ids=input_ids,
        attention_mask=attention_mask_half,
    )
    logits_half = outputs_half.logits[0, -1, :10]  # 最后一个token的前10个logits
    print(f"Last token logits (first 10): {logits_half.cpu()}")

# 比较输出
print("\nComparison:")
diff = (logits_full - logits_half).abs()
print(f"Absolute difference: {diff.cpu()}")
print(f"Max difference: {diff.max().item()}")
print(f"Mean difference: {diff.mean().item()}")

if diff.max().item() > 0.1:
    print("✓ Attention mask affects output (difference > 0.1)")
else:
    print("✗ WARNING: Attention mask does NOT significantly affect output!")

print("\n" + "=" * 60)
print("Test: Hook masked_fill_ 调用")
print("=" * 60)

# Hook masked_fill_ 来查看它是否被调用
# 注意：需要 hook 实际的 aten 操作，而不是 Python 方法
masked_fill_calls = []

# Hook 非 inplace 版本
original_masked_fill = torch.masked_fill

def hooked_masked_fill(self, mask, value):
    masked_fill_calls.append({
        'shape': self.shape,
        'mask_shape': mask.shape if hasattr(mask, 'shape') else None,
        'value': value,
        'device': str(self.device),
    })
    if len(masked_fill_calls) <= 5:
        print(f"[MASKED_FILL] shape={self.shape}, mask_shape={mask.shape if hasattr(mask, 'shape') else None}, value={value}")
    return original_masked_fill(self, mask, value)

torch.masked_fill = hooked_masked_fill

# 也 hook inplace 版本
original_masked_fill_ = torch.Tensor.masked_fill_

def hooked_masked_fill_(self, mask, value):
    masked_fill_calls.append({
        'shape': self.shape,
        'mask_shape': mask.shape if hasattr(mask, 'shape') else None,
        'value': value,
        'device': str(self.device),
        'inplace': True,
    })
    if len(masked_fill_calls) <= 5:
        print(f"[MASKED_FILL_] shape={self.shape}, mask_shape={mask.shape if hasattr(mask, 'shape') else None}, value={value}")
    return original_masked_fill_(self, mask, value)

torch.Tensor.masked_fill_ = hooked_masked_fill_

# 执行 forward pass
print("Executing forward pass (hooking masked_fill_)...")
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask_full,
    )

print(f"\nTotal masked_fill_ calls: {len(masked_fill_calls)}")
if len(masked_fill_calls) > 0:
    print("✓ masked_fill_ is being called")
    print(f"First few calls:")
    for i, call in enumerate(masked_fill_calls[:5]):
        print(f"  {i+1}. shape={call['shape']}, value={call['value']}")
else:
    print("✗ WARNING: masked_fill_ is NOT being called!")

# 恢复原始函数
torch.Tensor.masked_fill_ = original_masked_fill_
torch.masked_fill = original_masked_fill

print("\n✓ All tests completed!")

#!/usr/bin/env python
"""
测试生成过程中 attention_mask 的使用

检查生成过程中模型是否正确关注输入序列
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cas_npu
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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
print("Test: 检查生成过程中 attention_mask 的扩展")
print("=" * 60)

prompt = "讲个笑话"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs.get("attention_mask", None)
if attention_mask is not None:
    attention_mask = attention_mask.to(device)
else:
    attention_mask = torch.ones_like(input_ids).to(device)

print(f"Prompt: {prompt}")
print(f"Input IDs: {input_ids.cpu().tolist()}")
print(f"Initial attention mask: {attention_mask.cpu().tolist()}")

# Hook forward 来查看每次调用时的 attention_mask
forward_calls = []

original_forward = model.forward

def hooked_forward(*args, **kwargs):
    call_info = {
        'input_ids_shape': kwargs.get('input_ids', args[0] if args else None).shape if 'input_ids' in kwargs or args else None,
        'attention_mask_shape': kwargs.get('attention_mask', None).shape if 'attention_mask' in kwargs else None,
        'attention_mask_sum': kwargs.get('attention_mask', None).sum().item() if 'attention_mask' in kwargs else None,
    }
    forward_calls.append(call_info)
    
    if len(forward_calls) <= 5:
        print(f"[FORWARD #{len(forward_calls)}] input_ids={call_info['input_ids_shape']}, "
              f"attention_mask={call_info['attention_mask_shape']}, "
              f"mask_sum={call_info['attention_mask_sum']}")
    
    return original_forward(*args, **kwargs)

model.forward = hooked_forward

# 执行生成
print("\nGenerating with attention_mask...")
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        do_sample=False,  # 贪心解码
    )

print(f"\nTotal forward calls: {len(forward_calls)}")
print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
print(f"Generated only: {tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)}")

# 检查 attention_mask 是否正确扩展
print("\nChecking attention_mask expansion:")
for i, call in enumerate(forward_calls[:10]):
    print(f"  Call {i+1}: attention_mask sum = {call['attention_mask_sum']}")

# 恢复
model.forward = original_forward

print("\n" + "=" * 60)
print("Test: 比较有/无 attention_mask 的生成结果")
print("=" * 60)

# 测试 1: 有 attention_mask
print("\nWith attention_mask:")
with torch.no_grad():
    outputs_with = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,
        do_sample=False,
    )
text_with = tokenizer.decode(outputs_with[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"Generated: {text_with}")

# 测试 2: 无 attention_mask
print("\nWithout attention_mask:")
with torch.no_grad():
    outputs_without = model.generate(
        input_ids=input_ids,
        max_new_tokens=10,
        do_sample=False,
    )
text_without = tokenizer.decode(outputs_without[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"Generated: {text_without}")

if text_with == text_without:
    print("\n⚠ WARNING: Outputs are identical with/without attention_mask!")
else:
    print("\n✓ Outputs differ with/without attention_mask")

print("\n✓ All tests completed!")

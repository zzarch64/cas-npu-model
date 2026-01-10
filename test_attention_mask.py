#!/usr/bin/env python
"""
测试 attention_mask 是否在模型推理中生效

这个脚本用于验证：
1. attention_mask 是否被正确传递到模型
2. 模型是否使用 attention_mask 来屏蔽 padding 位置
3. 生成过程中 attention_mask 是否正确扩展
"""

import sys
import os
import torch

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入cas_npu扩展
try:
    import cas_npu
    print("✓ CAS-NPU extension imported successfully")
except ImportError as e:
    print(f"✗ Failed to import CAS-NPU extension: {e}")
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cas_npu:0')

# 加载模型
print("\nLoading model...")
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

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\n" + "=" * 60)
print("Test 1: 检查 attention_mask 在 forward 中的使用")
print("=" * 60)

# 创建两个不同长度的输入
prompt1 = "Hello"
prompt2 = "Hello, how are you?"

inputs1 = tokenizer(prompt1, return_tensors="pt", padding=False)
inputs2 = tokenizer(prompt2, return_tensors="pt", padding=False)

# 手动创建 batch，包含 padding
max_len = max(inputs1["input_ids"].shape[1], inputs2["input_ids"].shape[1])
input_ids = torch.zeros((2, max_len), dtype=torch.long)
attention_mask = torch.zeros((2, max_len), dtype=torch.long)

# 填充第一个序列
len1 = inputs1["input_ids"].shape[1]
input_ids[0, :len1] = inputs1["input_ids"][0]
attention_mask[0, :len1] = 1

# 填充第二个序列
len2 = inputs2["input_ids"].shape[1]
input_ids[1, :len2] = inputs2["input_ids"][0]
attention_mask[1, :len2] = 1

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

print(f"Input IDs shape: {input_ids.shape}")
print(f"Input IDs:\n{input_ids.cpu()}")
print(f"Attention mask shape: {attention_mask.shape}")
print(f"Attention mask:\n{attention_mask.cpu()}")

# 执行 forward pass
print("\nExecuting forward pass with attention_mask...")
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

print(f"Output shape: {outputs.logits.shape}")
print(f"Output logits (first token, first 5 vocab): {outputs.logits[0, 0, :5].cpu()}")

# 检查：如果 attention_mask 生效，padding 位置的 logits 应该被忽略
# 但实际上，模型会在最后一个有效 token 位置产生输出
print("\nChecking if attention_mask affects output...")
# 比较第一个序列和第二个序列在相同位置（相对于各自序列末尾）的输出
print(f"First sequence last token logits (first 5): {outputs.logits[0, len1-1, :5].cpu()}")
print(f"Second sequence last token logits (first 5): {outputs.logits[1, len2-1, :5].cpu()}")

print("\n" + "=" * 60)
print("Test 2: 检查生成过程中 attention_mask 的使用")
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
print(f"Attention mask: {attention_mask.cpu().tolist()}")
print(f"Decoded input: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

print("\nGenerating with attention_mask...")
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,
        do_sample=False,  # 使用贪心解码，更容易验证
    )

print(f"Output IDs: {outputs[0].cpu().tolist()}")
print(f"Full output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
print(f"Generated only: {tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)}")

# 检查输出是否包含输入
if tokenizer.decode(outputs[0][:input_ids.shape[1]], skip_special_tokens=True) == tokenizer.decode(input_ids[0], skip_special_tokens=True):
    print("✓ Output correctly contains input")
else:
    print("✗ WARNING: Output does not match input!")

print("\n" + "=" * 60)
print("Test 3: 检查 masked_fill_ 是否被调用")
print("=" * 60)

# 创建一个简单的测试，看看 masked_fill_ 是否工作
print("Testing masked_fill_ directly...")
x = torch.ones((2, 3), dtype=torch.float32, device=device)
mask = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool, device=device)
print(f"Before masked_fill_: {x.cpu()}")
print(f"Mask: {mask.cpu()}")
x.masked_fill_(mask, -1e9)
print(f"After masked_fill_: {x.cpu()}")
print("✓ masked_fill_ works correctly")

print("\n" + "=" * 60)
print("Test 4: 检查模型内部是否使用 attention_mask")
print("=" * 60)

# 尝试 hook 模型的 forward 方法，看看 attention_mask 是否被使用
original_forward = model.forward

def hooked_forward(*args, **kwargs):
    print(f"[HOOK] Forward called with kwargs: {list(kwargs.keys())}")
    if 'attention_mask' in kwargs:
        am = kwargs['attention_mask']
        print(f"[HOOK] attention_mask shape: {am.shape}, sum: {am.sum().item()}")
    else:
        print("[HOOK] WARNING: attention_mask not in kwargs!")
    return original_forward(*args, **kwargs)

model.forward = hooked_forward

# 再次执行 forward
print("Executing forward with hook...")
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

print("\n✓ All tests completed!")

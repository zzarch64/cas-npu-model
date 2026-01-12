#!/usr/bin/env python
"""
详细测试 _copy_from 的各种情况

检查数据在拷贝过程中是否被损坏
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cas_npu

device_npu = torch.device('cas_npu:0')

print("=" * 60)
print("Test 1: 基本拷贝测试")
print("=" * 60)

# 1.1 CPU -> NPU
print("\n1.1 CPU -> NPU...")
x_cpu = torch.randn(3, 4, dtype=torch.float32)
x_npu = x_cpu.to(device_npu)
x_back = x_npu.cpu()

diff = (x_cpu - x_back).abs()
print(f"  Max difference: {diff.max().item():.10f}")
if diff.max().item() > 1e-7:
    print(f"  ✗ CPU -> NPU -> CPU copy differs!")
    print(f"  Original: {x_cpu[0, :5]}")
    print(f"  After copy: {x_back[0, :5]}")
else:
    print(f"  ✓ CPU -> NPU -> CPU copy matches")

# 1.2 NPU -> CPU
print("\n1.2 NPU -> CPU...")
x_npu = torch.randn(3, 4, dtype=torch.float32, device=device_npu)
x_cpu = x_npu.cpu()
x_back = x_cpu.to(device_npu).cpu()

diff = (x_cpu - x_back).abs()
print(f"  Max difference: {diff.max().item():.10f}")
if diff.max().item() > 1e-7:
    print(f"  ✗ NPU -> CPU -> NPU copy differs!")
else:
    print(f"  ✓ NPU -> CPU -> NPU copy matches")

# 1.3 NPU -> NPU
print("\n1.3 NPU -> NPU...")
x_npu = torch.randn(3, 4, dtype=torch.float32, device=device_npu)
y_npu = x_npu.clone()
diff = (x_npu.cpu() - y_npu.cpu()).abs()
print(f"  Max difference: {diff.max().item():.10f}")
if diff.max().item() > 1e-7:
    print(f"  ✗ NPU -> NPU clone differs!")
else:
    print(f"  ✓ NPU -> NPU clone matches")

print("\n" + "=" * 60)
print("Test 2: 非 contiguous tensor 拷贝")
print("=" * 60)

# 2.1 Transpose (非 contiguous)
print("\n2.1 Transpose tensor...")
x_cpu = torch.randn(3, 4, dtype=torch.float32)
x_t_cpu = x_cpu.t()  # 非 contiguous
print(f"  Original is contiguous: {x_cpu.is_contiguous()}")
print(f"  Transposed is contiguous: {x_t_cpu.is_contiguous()}")

x_t_npu = x_t_cpu.to(device_npu)
x_t_back = x_t_npu.cpu()

diff = (x_t_cpu - x_t_back).abs()
print(f"  Max difference: {diff.max().item():.10f}")
if diff.max().item() > 1e-7:
    print(f"  ✗ Transpose copy differs!")
    print(f"  Original: {x_t_cpu[0, :]}")
    print(f"  After copy: {x_t_back[0, :]}")
else:
    print(f"  ✓ Transpose copy matches")

# 2.2 Slice (非 contiguous)
print("\n2.2 Sliced tensor...")
x_cpu = torch.randn(10, 10, dtype=torch.float32)
x_slice_cpu = x_cpu[::2, ::2]  # 非 contiguous
print(f"  Sliced is contiguous: {x_slice_cpu.is_contiguous()}")

x_slice_npu = x_slice_cpu.to(device_npu)
x_slice_back = x_slice_npu.cpu()

diff = (x_slice_cpu - x_slice_back).abs()
print(f"  Max difference: {diff.max().item():.10f}")
if diff.max().item() > 1e-7:
    print(f"  ✗ Slice copy differs!")
else:
    print(f"  ✓ Slice copy matches")

# 2.3 View/Reshape
print("\n2.3 Reshaped tensor...")
x_cpu = torch.randn(12, dtype=torch.float32)
x_view_cpu = x_cpu.view(3, 4)

x_view_npu = x_view_cpu.to(device_npu)
x_view_back = x_view_npu.cpu()

diff = (x_view_cpu - x_view_back).abs()
print(f"  Max difference: {diff.max().item():.10f}")
if diff.max().item() > 1e-7:
    print(f"  ✗ View copy differs!")
else:
    print(f"  ✓ View copy matches")

print("\n" + "=" * 60)
print("Test 3: 3D tensor (模拟 attention 输出)")
print("=" * 60)

# 3.1 3D contiguous
print("\n3.1 3D contiguous tensor...")
x_cpu = torch.randn(1, 3, 896, dtype=torch.float32)
x_npu = x_cpu.to(device_npu)
x_back = x_npu.cpu()

diff = (x_cpu - x_back).abs()
print(f"  Max difference: {diff.max().item():.10f}")
if diff.max().item() > 1e-7:
    print(f"  ✗ 3D copy differs!")
else:
    print(f"  ✓ 3D copy matches")

# 3.2 3D 非 contiguous (permute)
print("\n3.2 3D permuted tensor...")
x_cpu = torch.randn(1, 896, 3, dtype=torch.float32)
x_perm_cpu = x_cpu.permute(0, 2, 1)  # [1, 3, 896] 但非 contiguous
print(f"  Permuted is contiguous: {x_perm_cpu.is_contiguous()}")

x_perm_npu = x_perm_cpu.to(device_npu)
x_perm_back = x_perm_npu.cpu()

diff = (x_perm_cpu - x_perm_back).abs()
print(f"  Max difference: {diff.max().item():.10f}")
if diff.max().item() > 1e-7:
    print(f"  ✗ 3D permuted copy differs!")
    print(f"  Original shape: {x_perm_cpu.shape}, stride: {x_perm_cpu.stride()}")
    print(f"  After copy shape: {x_perm_back.shape}, stride: {x_perm_back.stride()}")
else:
    print(f"  ✓ 3D permuted copy matches")

print("\n" + "=" * 60)
print("Test 4: 模拟实际模型中的数据传递")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

prompt = "讲个笑话"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", None)

input_ids_npu = input_ids.to(device_npu)
attention_mask_npu = attention_mask.to(device_npu) if attention_mask is not None else None

# 获取 attention 输出
print("\n4.1 Comparing attention output directly...")
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
print(f"  CPU attention output is contiguous: {attn_out_cpu.is_contiguous()}")
print(f"  NPU attention output is contiguous: {attn_out_npu.is_contiguous()}")

attn_diff = (attn_out_cpu - attn_out_npu.cpu()).abs()
print(f"  Attention output max diff: {attn_diff.max().item():.6f}")

# 检查 NPU attention 输出拷贝回 CPU 后的数据
print("\n4.2 Copy NPU attention output to CPU and back...")
attn_out_npu_to_cpu = attn_out_npu.cpu()
attn_out_back_to_npu = attn_out_npu_to_cpu.to(device_npu)
attn_out_back_to_cpu = attn_out_back_to_npu.cpu()

copy_diff = (attn_out_npu_to_cpu - attn_out_back_to_cpu).abs()
print(f"  Copy round-trip max diff: {copy_diff.max().item():.10f}")
if copy_diff.max().item() > 1e-7:
    print(f"  ✗ Copy round-trip differs!")
else:
    print(f"  ✓ Copy round-trip matches")

# 检查 layer norm 输入
print("\n4.3 Layer norm input comparison...")
# 在 Qwen2 中，layer norm 在 attention 之后
# input_layernorm 用于 attention 输入，post_attention_layernorm 用于 FFN 输入

# 检查 post_attention_layernorm 的输入
print("  Checking post_attention_layernorm...")

# 使用 attention 输出作为 layer norm 的输入（模拟）
# 注意：实际上还有残差连接
ln_input_cpu = attn_out_cpu
ln_input_npu = attn_out_npu

ln_output_cpu = first_layer_cpu.post_attention_layernorm(ln_input_cpu)
ln_output_npu = first_layer_npu.post_attention_layernorm(ln_input_npu)

ln_diff = (ln_output_cpu - ln_output_npu.cpu()).abs()
print(f"  Layer norm output max diff: {ln_diff.max().item():.6f}")
print(f"  Layer norm output mean diff: {ln_diff.mean().item():.6f}")

if ln_diff.max().item() > 0.1:
    print(f"  ✗ Layer norm output differs significantly!")
    # 找出差异最大的位置
    max_idx = ln_diff.argmax()
    idx_0, idx_1, idx_2 = torch.unravel_index(max_idx, ln_diff.shape)
    print(f"  Max diff at [{idx_0}, {idx_1}, {idx_2}]")
    print(f"  CPU value: {ln_output_cpu[idx_0, idx_1, idx_2].item():.6f}")
    print(f"  NPU value: {ln_output_npu[idx_0, idx_1, idx_2].cpu().item():.6f}")
    print(f"  Input CPU value: {ln_input_cpu[idx_0, idx_1, idx_2].item():.6f}")
    print(f"  Input NPU value: {ln_input_npu[idx_0, idx_1, idx_2].cpu().item():.6f}")
else:
    print(f"  ✓ Layer norm output matches")

print("\n" + "=" * 60)
print("Test 5: 检查 layer norm 内部的数据传递")
print("=" * 60)

# 检查 layer norm 的 weight 和 bias 是否正确传递到 NPU
ln_weight_cpu = first_layer_cpu.post_attention_layernorm.weight.data
ln_weight_npu = first_layer_npu.post_attention_layernorm.weight.data.cpu()

ln_weight_diff = (ln_weight_cpu - ln_weight_npu).abs()
print(f"\n5.1 Layer norm weight comparison:")
print(f"  Max difference: {ln_weight_diff.max().item():.10f}")
if ln_weight_diff.max().item() > 1e-7:
    print(f"  ✗ Layer norm weight differs!")
else:
    print(f"  ✓ Layer norm weight matches")

# 检查 layer norm 计算
print("\n5.2 Manual layer norm calculation...")
# Layer norm: (x - mean) / sqrt(var + eps) * weight + bias
eps = first_layer_cpu.post_attention_layernorm.eps
normalized_shape = first_layer_cpu.post_attention_layernorm.normalized_shape

# 使用相同的输入（CPU attention 输出）进行 layer norm
ln_input_same = attn_out_cpu
ln_input_same_npu = ln_input_same.to(device_npu)

# CPU 计算
ln_out_cpu_manual = first_layer_cpu.post_attention_layernorm(ln_input_same)

# NPU 计算
ln_out_npu_manual = first_layer_npu.post_attention_layernorm(ln_input_same_npu)

ln_manual_diff = (ln_out_cpu_manual - ln_out_npu_manual.cpu()).abs()
print(f"  Using same input (CPU attention output):")
print(f"  Max difference: {ln_manual_diff.max().item():.6f}")
print(f"  Mean difference: {ln_manual_diff.mean().item():.6f}")

if ln_manual_diff.max().item() > 0.1:
    print(f"  ✗ Layer norm with same input differs!")
    # 找出差异最大的位置
    max_idx = ln_manual_diff.argmax()
    idx_0, idx_1, idx_2 = torch.unravel_index(max_idx, ln_manual_diff.shape)
    print(f"  Max diff at [{idx_0}, {idx_1}, {idx_2}]")
    print(f"  CPU output: {ln_out_cpu_manual[idx_0, idx_1, idx_2].item():.6f}")
    print(f"  NPU output: {ln_out_npu_manual[idx_0, idx_1, idx_2].cpu().item():.6f}")
else:
    print(f"  ✓ Layer norm with same input matches")

print("\n✓ All tests completed!")

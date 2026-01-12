#!/usr/bin/env python
"""
逐步检查 FFN 的每个计算步骤

测试内容:
1. Input layer norm
2. Gate projection
3. Up projection
4. SiLU activation
5. Multiply (SiLU(gate) * up)
6. Down projection
7. Complete FFN output

使用方法:
    python test/integration/model/test_ffn_step_by_step.py
    python test/integration/model/test_ffn_step_by_step.py -vv
    python test/integration/model/test_ffn_step_by_step.py --model-path Qwen/Qwen2.5-0.5B
"""

import sys
import os
import argparse

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导入测试框架
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from test_framework import (
    ensure_cas_npu, TestConfig, VerbosityLevel, check_tensor, verify_tensor_match,
    print_section, print_step, create_arg_parser, run_test
)

import torch


def run_ffn_step_by_step_test(config: TestConfig, model_path: str) -> bool:
    """运行 FFN 逐步测试"""
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ⚠ transformers not installed, skipping test")
        return True
    
    device = torch.device(config.device)
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    
    model_cpu = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32, local_files_only=True
    )
    model_cpu.eval()
    
    model_npu = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32, local_files_only=True
    )
    model_npu = model_npu.to(device)
    model_npu.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "讲个笑话"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    input_ids_npu = input_ids.to(device)
    attention_mask_npu = attention_mask.to(device) if attention_mask is not None else None
    
    # 获取第一层的输入（attention 输出）
    print_section("Getting Attention Output", config)
    first_layer_cpu = model_cpu.model.layers[0]
    first_layer_npu = model_npu.model.layers[0]
    
    attn_out_cpu = None
    attn_out_npu = None
    
    def hook_attn_cpu(module, input, output):
        nonlocal attn_out_cpu
        attn_out_cpu = output[0].detach().clone()
    
    def hook_attn_npu(module, input, output):
        nonlocal attn_out_npu
        attn_out_npu = output[0].detach().clone()
    
    handle_attn_cpu = first_layer_cpu.self_attn.register_forward_hook(hook_attn_cpu)
    handle_attn_npu = first_layer_npu.self_attn.register_forward_hook(hook_attn_npu)
    
    with torch.no_grad():
        _ = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
        _ = model_npu(input_ids=input_ids_npu, attention_mask=attention_mask_npu)
    
    handle_attn_cpu.remove()
    handle_attn_npu.remove()
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Attention output shape: {attn_out_cpu.shape}")
    
    matched, _ = verify_tensor_match(
        attn_out_npu.cpu(), attn_out_cpu, "attention output", config.tolerance, config
    )
    
    all_passed = True
    if not matched:
        all_passed = False
    
    # 逐步检查 FFN
    print_section("Step-by-step FFN Check", config)
    
    # 1. Input layer norm
    print_step("1. Input layer norm", config)
    input_norm_cpu = first_layer_cpu.input_layernorm(attn_out_cpu)
    input_norm_npu = first_layer_npu.input_layernorm(attn_out_npu)
    
    matched, _ = verify_tensor_match(
        input_norm_npu.cpu(), input_norm_cpu, "input layer norm", config.tolerance, config
    )
    if not matched:
        all_passed = False
    
    # FFN: gate_proj, up_proj, SiLU, multiply, down_proj
    ffn_cpu = first_layer_cpu.mlp
    ffn_npu = first_layer_npu.mlp
    
    # 2. Gate projection
    print_step("2. Gate projection", config)
    gate_out_cpu = ffn_cpu.gate_proj(input_norm_cpu)
    gate_out_npu = ffn_npu.gate_proj(input_norm_npu).cpu()
    
    matched, _ = verify_tensor_match(gate_out_npu, gate_out_cpu, "gate projection", config.tolerance, config)
    if not matched:
        all_passed = False
    
    # 3. Up projection
    print_step("3. Up projection", config)
    up_out_cpu = ffn_cpu.up_proj(input_norm_cpu)
    up_out_npu = ffn_npu.up_proj(input_norm_npu).cpu()
    
    matched, _ = verify_tensor_match(up_out_npu, up_out_cpu, "up projection", config.tolerance, config)
    if not matched:
        all_passed = False
    
    # 4. SiLU activation
    print_step("4. SiLU activation (on gate)", config)
    silu_gate_cpu = torch.nn.functional.silu(gate_out_cpu)
    silu_gate_npu = torch.nn.functional.silu(gate_out_npu.to(device)).cpu()
    
    matched, _ = verify_tensor_match(silu_gate_npu, silu_gate_cpu, "SiLU activation", config.tolerance, config)
    if not matched:
        all_passed = False
    
    # 5. Multiply
    print_step("5. Multiply (SiLU(gate) * up)", config)
    multiply_cpu = silu_gate_cpu * up_out_cpu
    multiply_npu = silu_gate_npu * up_out_npu
    
    matched, _ = verify_tensor_match(multiply_npu, multiply_cpu, "multiply", config.tolerance, config)
    if not matched:
        all_passed = False
    
    # 6. Down projection
    print_step("6. Down projection", config)
    down_out_cpu = ffn_cpu.down_proj(multiply_cpu)
    down_out_npu = ffn_npu.down_proj(multiply_npu.to(device)).cpu()
    
    matched, _ = verify_tensor_match(down_out_npu, down_out_cpu, "down projection", config.tolerance, config)
    if not matched:
        all_passed = False
    
    # 7. Complete FFN output
    print_step("7. Complete FFN output", config)
    ffn_out_cpu = ffn_cpu(input_norm_cpu)
    ffn_out_npu = ffn_npu(input_norm_npu).cpu()
    
    matched, _ = verify_tensor_match(ffn_out_npu, ffn_out_cpu, "complete FFN output", config.tolerance, config)
    if not matched:
        all_passed = False
    
    return all_passed


def main():
    parser = create_arg_parser("FFN Step-by-Step Test")
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen2.5-0.5B',
        help='Model path (default: Qwen/Qwen2.5-0.5B)'
    )
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_cas_npu()
    
    print_section("FFN Step-by-Step Test", config)
    
    success = run_ffn_step_by_step_test(config, args.model_path)
    
    print_section("Test Summary", config)
    if success:
        print("All steps passed! ✓")
    else:
        print("Some steps failed! ✗")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

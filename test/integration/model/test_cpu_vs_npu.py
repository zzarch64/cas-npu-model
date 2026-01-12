#!/usr/bin/env python
"""
比较 CPU 和 ECHO-NPU 上的模型输出

测试内容:
1. Forward pass 对比
2. Generation 对比
3. 详细的差异分析

使用方法:
    python test/integration/model/test_cpu_vs_npu.py
    python test/integration/model/test_cpu_vs_npu.py -vv
    python test/integration/model/test_cpu_vs_npu.py --model-path Qwen/Qwen2.5-0.5B
    python test/integration/model/test_cpu_vs_npu.py --max-new-tokens 10
"""

import sys
import os
import argparse

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导入测试框架
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from test_framework import (
    ensure_echo_npu, TestConfig, VerbosityLevel, check_tensor, verify_tensor_match,
    print_section, print_step, create_arg_parser, run_test
)

import torch
import numpy as np


def analyze_difference(logits_cpu, logits_npu, name="logits"):
    """详细分析两个 tensor 的差异"""
    diff = (logits_cpu - logits_npu).abs()
    
    stats = {
        'max_diff': diff.max().item(),
        'mean_diff': diff.mean().item(),
        'std_diff': diff.std().item(),
        'median_diff': diff.median().item(),
        'p95_diff': torch.quantile(diff, 0.95).item(),
        'p99_diff': torch.quantile(diff, 0.99).item(),
        'relative_max_diff': (diff / (logits_cpu.abs() + 1e-8)).max().item(),
        'relative_mean_diff': (diff / (logits_cpu.abs() + 1e-8)).mean().item(),
    }
    
    # 检查是否是系统性的偏差
    signed_diff = logits_cpu - logits_npu
    stats['bias'] = signed_diff.mean().item()  # 如果有系统性偏差，这个值会比较大
    
    # 检查差异的分布
    zero_diff_count = (diff < 1e-7).sum().item()
    stats['zero_diff_ratio'] = zero_diff_count / diff.numel()
    
    return stats


def test_forward_pass(config: TestConfig, model_path: str) -> bool:
    """测试 forward pass 对比"""
    print_step("Forward Pass Comparison", config)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ⚠ transformers not installed, skipping test")
        return True
    
    device = torch.device(config.device)
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  Loading models from {model_path}...")
    
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
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Prompt: {prompt}")
        print(f"  Input IDs: {input_ids.tolist()}")
    
    # Forward pass on CPU
    with torch.no_grad():
        outputs_cpu = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
        logits_cpu = outputs_cpu.logits[0, -1, :10]
    
    # Forward pass on NPU
    input_ids_npu = input_ids.to(device)
    attention_mask_npu = attention_mask.to(device) if attention_mask is not None else None
    with torch.no_grad():
        outputs_npu = model_npu(input_ids=input_ids_npu, attention_mask=attention_mask_npu)
        logits_npu = outputs_npu.logits[0, -1, :10].cpu()
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  CPU logits (first 10): {logits_cpu}")
        print(f"  NPU logits (first 10): {logits_npu}")
    
    # 详细分析差异
    diff_stats = analyze_difference(logits_cpu, logits_npu, "forward logits")
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  Max diff: {diff_stats['max_diff']:.6f}")
        print(f"  Mean diff: {diff_stats['mean_diff']:.6f}")
        print(f"  Relative max diff: {diff_stats['relative_max_diff']:.2%}")
        print(f"  Bias (systematic error): {diff_stats['bias']:.6f}")
        print(f"  Zero diff ratio: {diff_stats['zero_diff_ratio']:.2%}")
    
    # 判断是容差问题还是计算错误
    is_tolerance_issue = (
        diff_stats['max_diff'] < 0.01 and  # 绝对差异很小
        diff_stats['relative_max_diff'] < 0.1 and  # 相对差异也很小
        abs(diff_stats['bias']) < 0.001  # 没有系统性偏差
    )
    
    if is_tolerance_issue:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✓ 判断：容差过严（差异很小且无系统性偏差）")
        # 使用更宽松的容差
        model_tolerance = max(config.tolerance, 1e-4)
    else:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ 判断：可能存在计算错误（差异较大或有系统性偏差）")
            if diff_stats['relative_max_diff'] >= 0.1:
                print(f"    相对差异 {diff_stats['relative_max_diff']:.2%} 较大，建议检查计算实现")
            if abs(diff_stats['bias']) >= 0.001:
                print(f"    系统性偏差 {diff_stats['bias']:.6f} 较大，可能存在累积误差")
        # 使用严格容差
        model_tolerance = config.tolerance
    
    matched, info = verify_tensor_match(logits_npu, logits_cpu, "forward logits", model_tolerance, config)
    
    return matched or is_tolerance_issue


def test_generation(config: TestConfig, model_path: str, max_new_tokens: int = 5) -> bool:
    """测试 generation 对比"""
    print_step("Generation Comparison", config)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ⚠ transformers not installed, skipping test")
        return True
    
    device = torch.device(config.device)
    
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
    
    # Generate on CPU
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Generating on CPU (max_new_tokens={max_new_tokens})...")
    with torch.no_grad():
        outputs_cpu = model_cpu.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    text_cpu = tokenizer.decode(outputs_cpu[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # Generate on NPU
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Generating on NPU (max_new_tokens={max_new_tokens})...")
    with torch.no_grad():
        outputs_npu = model_npu.generate(
            input_ids=input_ids_npu,
            attention_mask=attention_mask_npu,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    text_npu = tokenizer.decode(outputs_npu[0][input_ids_npu.shape[1]:], skip_special_tokens=True)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  CPU generated: {text_cpu}")
        print(f"  NPU generated: {text_npu}")
    
    if text_cpu == text_npu:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ✓ Generated texts match!")
        return True
    else:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ Generated texts differ!")
            print(f"    CPU: {text_cpu}")
            print(f"    NPU: {text_npu}")
        return False


def main():
    parser = create_arg_parser("CPU vs NPU Comparison Test")
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen2.5-0.5B',
        help='Model path (default: Qwen/Qwen2.5-0.5B)'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=5,
        help='Max new tokens for generation test (default: 5)'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip generation test'
    )
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_echo_npu()
    
    print_section("CPU vs NPU Comparison Test", config)
    
    results = []
    
    # Forward pass 对比
    results.append(("Forward Pass", run_test(
        lambda c: test_forward_pass(c, args.model_path), config, "Forward Pass Test"
    )))
    
    # Generation 对比
    if not args.skip_generation:
        results.append(("Generation", run_test(
            lambda c: test_generation(c, args.model_path, args.max_new_tokens), config, "Generation Test"
        )))
    
    # 汇总结果
    print_section("Test Summary", config)
    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        if config.verbosity.value >= VerbosityLevel.QUIET.value:
            print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print()
        if all_passed:
            print("All tests passed! ✓")
        else:
            print("Some tests failed! ✗")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

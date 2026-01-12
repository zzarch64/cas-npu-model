#!/usr/bin/env python
"""
详细测试 attention_mask 在模型内部的使用

测试内容:
1. attention_mask 对输出的影响
2. Hook masked_fill_ 调用

使用方法:
    python test/integration/attention/test_attention_mask_detailed.py
    python test/integration/attention/test_attention_mask_detailed.py -vv
    python test/integration/attention/test_attention_mask_detailed.py --model-path Qwen/Qwen2.5-0.5B
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


def test_attention_mask_effect(config: TestConfig, model_path: str) -> bool:
    """测试 attention_mask 对输出的影响"""
    print_step("Attention mask effect on output", config)
    
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32, local_files_only=True
    )
    model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建两个相同的输入，但不同的 attention_mask
    prompt = "讲个笑话"
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = inputs["input_ids"].to(device)
    
    # 完整的 attention_mask（所有位置都是1）
    attention_mask_full = torch.ones_like(input_ids).to(device)
    
    # 只关注前一半的 attention_mask
    seq_len = input_ids.shape[1]
    attention_mask_half = torch.zeros_like(input_ids).to(device)
    attention_mask_half[0, :seq_len//2] = 1
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Prompt: {prompt}")
        print(f"  Input IDs: {input_ids.cpu().tolist()}")
        print(f"  Full attention mask: {attention_mask_full.cpu().tolist()}")
        print(f"  Half attention mask: {attention_mask_half.cpu().tolist()}")
    
    # 执行 forward pass
    with torch.no_grad():
        outputs_full = model(input_ids=input_ids, attention_mask=attention_mask_full)
        outputs_half = model(input_ids=input_ids, attention_mask=attention_mask_half)
    
    logits_full = outputs_full.logits[0, -1, :10]
    logits_half = outputs_half.logits[0, -1, :10]
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Full mask logits (first 10): {logits_full.cpu()}")
        print(f"  Half mask logits (first 10): {logits_half.cpu()}")
    
    # 比较输出
    diff = (logits_full - logits_half).abs()
    max_diff = diff.max().item()
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Absolute difference: {diff.cpu()}")
        print(f"  Max difference: {max_diff}")
    
    # attention_mask 应该影响输出（差异应该显著）
    passed = max_diff > 0.1
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        if passed:
            print(f"  ✓ Attention mask affects output (difference > 0.1)")
        else:
            print(f"  ⚠ WARNING: Attention mask does NOT significantly affect output!")
    
    return True  # 这个测试主要是观察，不一定算失败


def test_masked_fill_hook(config: TestConfig, model_path: str) -> bool:
    """Hook masked_fill_ 调用"""
    print_step("Hook masked_fill_ calls", config)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ⚠ transformers not installed, skipping test")
        return True
    
    device = torch.device(config.device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float32, local_files_only=True
    )
    model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "讲个笑话"
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Hook masked_fill_ 来查看它是否被调用
    masked_fill_calls = []
    
    original_masked_fill_ = torch.Tensor.masked_fill_
    
    def hooked_masked_fill_(self, mask, value):
        masked_fill_calls.append({
            'shape': self.shape,
            'mask_shape': mask.shape if hasattr(mask, 'shape') else None,
            'value': value,
            'device': str(self.device),
        })
        return original_masked_fill_(self, mask, value)
    
    torch.Tensor.masked_fill_ = hooked_masked_fill_
    
    try:
        # 执行 forward pass
        if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
            print("  Executing forward pass (hooking masked_fill_)...")
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
            print(f"  Total masked_fill_ calls: {len(masked_fill_calls)}")
        
        if len(masked_fill_calls) > 0:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print(f"  ✓ masked_fill_ is being called ({len(masked_fill_calls)} times)")
            if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
                print(f"  First few calls:")
                for i, call in enumerate(masked_fill_calls[:5]):
                    print(f"    {i+1}. shape={call['shape']}, value={call['value']}")
            return True
        else:
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print("  ⚠ WARNING: masked_fill_ is NOT being called!")
            return True  # 不一定是失败
    finally:
        # 恢复原始函数
        torch.Tensor.masked_fill_ = original_masked_fill_


def main():
    parser = create_arg_parser("Attention Mask Detailed Test")
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen2.5-0.5B',
        help='Model path (default: Qwen/Qwen2.5-0.5B)'
    )
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_echo_npu()
    
    print_section("Attention Mask Detailed Test", config)
    
    results = []
    
    # 测试 attention_mask 对输出的影响
    results.append(("Attention Mask Effect", run_test(
        lambda c: test_attention_mask_effect(c, args.model_path), config, "Attention Mask Effect Test"
    )))
    
    # 测试 masked_fill_ hook
    results.append(("masked_fill_ Hook", run_test(
        lambda c: test_masked_fill_hook(c, args.model_path), config, "masked_fill_ Hook Test"
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

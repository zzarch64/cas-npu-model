#!/usr/bin/env python
"""
测试 attention_mask 是否在模型推理中生效

测试内容:
1. Forward pass 中 attention_mask 的使用
2. Generation 中 attention_mask 的使用
3. masked_fill_ 操作测试

使用方法:
    python test/integration/attention/test_attention_mask.py
    python test/integration/attention/test_attention_mask.py -vv
    python test/integration/attention/test_attention_mask.py --model-path Qwen/Qwen2.5-0.5B
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


def test_masked_fill(config: TestConfig) -> bool:
    """测试 masked_fill_ 操作"""
    print_step("masked_fill_ operation", config)
    
    device = torch.device(config.device)
    
    x = torch.ones((2, 3), dtype=torch.float32, device=device)
    mask = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool, device=device)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Before masked_fill_: {x.cpu()}")
        print(f"  Mask: {mask.cpu()}")
    
    x.masked_fill_(mask, -1e9)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  After masked_fill_: {x.cpu()}")
    
    # 验证结果
    expected = torch.tensor([[-1e9, 1.0, -1e9], [1.0, -1e9, 1.0]], dtype=torch.float32)
    matched, info = verify_tensor_match(x.cpu(), expected, "masked_fill_", 1e-7, config)
    
    return matched


def test_attention_mask_forward(config: TestConfig, model_path: str) -> bool:
    """测试 forward pass 中 attention_mask 的使用"""
    print_step("Attention mask in forward pass", config)
    
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
    
    # 创建两个不同长度的输入
    prompt1 = "Hello"
    prompt2 = "Hello, how are you?"
    
    inputs1 = tokenizer(prompt1, return_tensors="pt", padding=False)
    inputs2 = tokenizer(prompt2, return_tensors="pt", padding=False)
    
    # 手动创建 batch，包含 padding
    max_len = max(inputs1["input_ids"].shape[1], inputs2["input_ids"].shape[1])
    input_ids = torch.zeros((2, max_len), dtype=torch.long)
    attention_mask = torch.zeros((2, max_len), dtype=torch.long)
    
    len1 = inputs1["input_ids"].shape[1]
    input_ids[0, :len1] = inputs1["input_ids"][0]
    attention_mask[0, :len1] = 1
    
    len2 = inputs2["input_ids"].shape[1]
    input_ids[1, :len2] = inputs2["input_ids"][0]
    attention_mask[1, :len2] = 1
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Input IDs:\n{input_ids.cpu()}")
        print(f"  Attention mask:\n{attention_mask.cpu()}")
    
    # 执行 forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Output shape: {outputs.logits.shape}")
        print(f"  First sequence last token logits (first 5): {outputs.logits[0, len1-1, :5].cpu()}")
        print(f"  Second sequence last token logits (first 5): {outputs.logits[1, len2-1, :5].cpu()}")
    
    # 验证输出形状
    passed = outputs.logits.shape == (2, max_len, model.config.vocab_size)
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        if passed:
            print("  ✓ Forward pass with attention_mask completed successfully")
        else:
            print("  ✗ Output shape mismatch")
    
    return passed


def test_attention_mask_generation(config: TestConfig, model_path: str) -> bool:
    """测试 generation 中 attention_mask 的使用"""
    print_step("Attention mask in generation", config)
    
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
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = torch.ones_like(input_ids).to(device)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Prompt: {prompt}")
        print(f"  Input IDs: {input_ids.cpu().tolist()}")
        print(f"  Attention mask: {attention_mask.cpu().tolist()}")
    
    # 执行 generation
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            do_sample=False,
        )
    
    generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Output IDs: {outputs[0].cpu().tolist()}")
        print(f"  Generated: {generated}")
    
    # 验证输出包含输入
    input_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    output_decoded = tokenizer.decode(outputs[0][:input_ids.shape[1]], skip_special_tokens=True)
    
    passed = input_decoded == output_decoded
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        if passed:
            print("  ✓ Generation with attention_mask completed successfully")
        else:
            print(f"  ✗ Output does not contain input correctly")
            print(f"    Input: {input_decoded}")
            print(f"    Output start: {output_decoded}")
    
    return passed


def main():
    parser = create_arg_parser("Attention Mask Test")
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen2.5-0.5B',
        help='Model path (default: Qwen/Qwen2.5-0.5B)'
    )
    parser.add_argument(
        '--skip-model-tests',
        action='store_true',
        help='Skip tests that require loading models'
    )
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_echo_npu()
    
    print_section("Attention Mask Test", config)
    
    results = []
    
    # masked_fill_ 测试
    results.append(("masked_fill_", run_test(test_masked_fill, config, "masked_fill_ Test")))
    
    # 模型测试
    if not args.skip_model_tests:
        print_section("Model Attention Mask Tests", config)
        results.append(("Forward Pass", run_test(
            lambda c: test_attention_mask_forward(c, args.model_path), config, "Forward Pass Test"
        )))
        results.append(("Generation", run_test(
            lambda c: test_attention_mask_generation(c, args.model_path), config, "Generation Test"
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

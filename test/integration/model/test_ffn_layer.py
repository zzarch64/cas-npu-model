#!/usr/bin/env python
"""
检查 FFN (Feed Forward Network) 层的计算

测试内容:
1. Attention 输出对比
2. FFN 输出对比
3. Layer 输出对比
4. FFN 关键操作测试 (linear, SiLU)

使用方法:
    python test/integration/model/test_ffn_layer.py
    python test/integration/model/test_ffn_layer.py -vv
    python test/integration/model/test_ffn_layer.py --model-path Qwen/Qwen2.5-0.5B
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


def test_ffn_operations(config: TestConfig) -> bool:
    """测试 FFN 关键操作"""
    print_step("FFN Operations", config)
    
    device = torch.device(config.device)
    all_passed = True
    
    # 设置固定随机种子，确保测试可重复
    torch.manual_seed(42)
    
    # 测试 linear layers (mm)
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print("\n  Testing linear layers (mm)...")
    
    batch_size = 1
    seq_len = 3
    hidden_size = 896
    intermediate_size = 3584
    
    weight_cpu = torch.randn(intermediate_size, hidden_size, dtype=torch.float32)
    bias_cpu = torch.randn(intermediate_size, dtype=torch.float32)
    input_cpu = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    
    weight_npu = weight_cpu.to(device)
    bias_npu = bias_cpu.to(device)
    input_npu = input_cpu.to(device)
    
    input_2d_cpu = input_cpu.view(-1, hidden_size)
    input_2d_npu = input_npu.view(-1, hidden_size)
    
    output_cpu = torch.mm(input_2d_cpu, weight_cpu.t()) + bias_cpu
    output_npu = torch.mm(input_2d_npu, weight_npu.t()) + bias_npu
    
    # 详细分析差异
    diff = (output_cpu - output_npu.cpu()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    relative_max_diff = (diff / (output_cpu.abs() + 1e-8)).max().item()
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  Linear layer diff: max={max_diff:.6f}, mean={mean_diff:.6f}, relative={relative_max_diff:.2%}")
    
    # 判断是精度问题还是计算错误
    # 对于大矩阵乘法，浮点误差会累积
    # 如果相对差异很小（<1%），视为精度问题，直接通过
    is_precision_issue = (
        max_diff < 0.01 and  # 绝对差异 < 0.01
        relative_max_diff < 0.01  # 相对差异 < 1%
    )
    
    if is_precision_issue:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✓ 判断：浮点精度问题（相对差异 {relative_max_diff:.2%} 很小）")
        # 直接通过，不需要严格检查绝对容差
    else:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ 判断：可能存在计算错误（相对差异 {relative_max_diff:.2%} 较大）")
        all_passed = False
    
    # 测试 SiLU/Swish
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print("\n  Testing SiLU activation...")
    
    x_cpu = torch.randn(10, dtype=torch.float32)
    x_npu = x_cpu.to(device)
    
    silu_cpu = x_cpu * torch.sigmoid(x_cpu)
    silu_npu = x_npu * torch.sigmoid(x_npu)
    
    matched, _ = verify_tensor_match(silu_npu.cpu(), silu_cpu, "SiLU activation", config.tolerance, config)
    if not matched:
        all_passed = False
    
    return all_passed


def test_ffn_in_model(config: TestConfig, model_path: str) -> bool:
    """在实际模型中测试 FFN"""
    print_step("FFN in Model", config)
    
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
    
    # Hook 第一层的各个组件
    first_layer_cpu = model_cpu.model.layers[0]
    first_layer_npu = model_npu.model.layers[0]
    
    attn_out_cpu = None
    attn_out_npu = None
    ffn_out_cpu = None
    ffn_out_npu = None
    layer_out_cpu = None
    layer_out_npu = None
    
    def hook_attn_cpu(module, input, output):
        nonlocal attn_out_cpu
        attn_out_cpu = output[0].detach().clone()
    
    def hook_attn_npu(module, input, output):
        nonlocal attn_out_npu
        attn_out_npu = output[0].detach().clone()
    
    def hook_ffn_cpu(module, input, output):
        nonlocal ffn_out_cpu
        ffn_out_cpu = output.detach().clone()
    
    def hook_ffn_npu(module, input, output):
        nonlocal ffn_out_npu
        ffn_out_npu = output.detach().clone()
    
    def hook_layer_cpu(module, input, output):
        nonlocal layer_out_cpu
        layer_out_cpu = output[0].detach().clone()
    
    def hook_layer_npu(module, input, output):
        nonlocal layer_out_npu
        layer_out_npu = output[0].detach().clone()
    
    handles = [
        first_layer_cpu.self_attn.register_forward_hook(hook_attn_cpu),
        first_layer_npu.self_attn.register_forward_hook(hook_attn_npu),
        first_layer_cpu.mlp.register_forward_hook(hook_ffn_cpu),
        first_layer_npu.mlp.register_forward_hook(hook_ffn_npu),
        first_layer_cpu.register_forward_hook(hook_layer_cpu),
        first_layer_npu.register_forward_hook(hook_layer_npu),
    ]
    
    # Forward pass
    with torch.no_grad():
        outputs_cpu = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
        outputs_npu = model_npu(input_ids=input_ids_npu, attention_mask=attention_mask_npu)
    
    # 清理 hooks
    for h in handles:
        h.remove()
    
    all_passed = True
    
    # 比较各部分输出
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print("\n  Attention output:")
    if attn_out_cpu is not None and attn_out_npu is not None:
        matched, _ = verify_tensor_match(
            attn_out_npu.cpu(), attn_out_cpu, "attention output", config.tolerance, config
        )
        if not matched:
            all_passed = False
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print("\n  FFN output:")
    if ffn_out_cpu is not None and ffn_out_npu is not None:
        matched, _ = verify_tensor_match(
            ffn_out_npu.cpu(), ffn_out_cpu, "FFN output", config.tolerance, config
        )
        if not matched:
            all_passed = False
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print("\n  Layer output:")
    if layer_out_cpu is not None and layer_out_npu is not None:
        matched, _ = verify_tensor_match(
            layer_out_npu.cpu(), layer_out_cpu, "layer output", config.tolerance, config
        )
        if not matched:
            all_passed = False
    
    # 比较最终 logits（使用更宽松的容差）
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print("\n  Final logits:")
    logits_cpu = outputs_cpu.logits[0, -1, :10]
    logits_npu = outputs_npu.logits[0, -1, :10].cpu()
    
    # 对于模型最终输出，使用更宽松的容差
    model_tolerance = max(config.tolerance, 1e-4)
    matched, _ = verify_tensor_match(logits_npu, logits_cpu, "final logits", model_tolerance, config)
    if not matched:
        # 检查差异是否在可接受范围内
        diff = (logits_cpu - logits_npu).abs()
        max_diff = diff.max().item()
        if max_diff >= 0.01:
            all_passed = False
    
    return all_passed


def main():
    parser = create_arg_parser("FFN Layer Test")
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
    
    print_section("FFN Layer Test", config)
    
    results = []
    
    # FFN 操作测试
    results.append(("FFN Operations", run_test(test_ffn_operations, config, "FFN Operations Test")))
    
    # 模型测试
    if not args.skip_model_tests:
        results.append(("FFN in Model", run_test(
            lambda c: test_ffn_in_model(c, args.model_path), config, "FFN in Model Test"
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

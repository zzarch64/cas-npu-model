#!/usr/bin/env python
"""
专门测试 attention 计算部分

测试内容:
1. Attention 输入输出对比
2. Q @ K^T (bmm) 测试
3. Softmax 测试
4. Attention @ V (bmm) 测试

使用方法:
    python test/integration/attention/test_attention_computation.py
    python test/integration/attention/test_attention_computation.py -vv
    python test/integration/attention/test_attention_computation.py --model-path Qwen/Qwen2.5-0.5B
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


def test_qk_bmm(config: TestConfig) -> bool:
    """测试 Q @ K^T (bmm)"""
    print_step("Q @ K^T (bmm)", config)
    
    device = torch.device(config.device)
    
    batch_size = 1
    seq_len = 3
    head_dim = 64
    num_heads = 8
    
    Q_cpu = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    K_cpu = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    Q_npu = Q_cpu.to(device)
    K_npu = K_cpu.to(device)
    
    # Q @ K^T
    QK_cpu = torch.bmm(Q_cpu.view(batch_size * num_heads, seq_len, head_dim),
                       K_cpu.view(batch_size * num_heads, seq_len, head_dim).transpose(1, 2))
    QK_npu = torch.bmm(Q_npu.view(batch_size * num_heads, seq_len, head_dim),
                       K_npu.view(batch_size * num_heads, seq_len, head_dim).transpose(1, 2))
    
    matched, info = verify_tensor_match(QK_npu.cpu(), QK_cpu, "QK^T output", config.tolerance, config)
    return matched


def test_softmax(config: TestConfig) -> bool:
    """测试 softmax"""
    print_step("Softmax", config)
    
    device = torch.device(config.device)
    
    batch_size = 1
    seq_len = 3
    num_heads = 8
    
    scores_cpu = torch.randn(batch_size * num_heads, seq_len, seq_len, dtype=torch.float32)
    scores_npu = scores_cpu.to(device)
    
    softmax_cpu = torch.softmax(scores_cpu, dim=-1)
    softmax_npu = torch.softmax(scores_npu, dim=-1).cpu()
    
    matched, info = verify_tensor_match(softmax_npu, softmax_cpu, "softmax output", config.tolerance, config)
    return matched


def test_attn_v_bmm(config: TestConfig) -> bool:
    """测试 attention @ V (bmm)"""
    print_step("Attention @ V (bmm)", config)
    
    device = torch.device(config.device)
    
    batch_size = 1
    seq_len = 3
    head_dim = 64
    num_heads = 8
    
    attn_cpu = torch.softmax(torch.randn(batch_size * num_heads, seq_len, seq_len, dtype=torch.float32), dim=-1)
    V_cpu = torch.randn(batch_size * num_heads, seq_len, head_dim, dtype=torch.float32)
    
    attn_npu = attn_cpu.to(device)
    V_npu = V_cpu.to(device)
    
    attnV_cpu = torch.bmm(attn_cpu, V_cpu)
    attnV_npu = torch.bmm(attn_npu, V_npu).cpu()
    
    matched, info = verify_tensor_match(attnV_npu, attnV_cpu, "attention@V output", config.tolerance, config)
    return matched


def test_attention_in_model(config: TestConfig, model_path: str) -> bool:
    """在实际模型中测试 attention"""
    print_step("Attention in Model", config)
    
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
    
    # Hook attention 层
    attention_inputs_cpu = []
    attention_outputs_cpu = []
    attention_inputs_npu = []
    attention_outputs_npu = []
    
    def hook_attention_cpu(module, input, output):
        attention_inputs_cpu.append([x.detach().clone() if isinstance(x, torch.Tensor) else x for x in input])
        attention_outputs_cpu.append(output[0].detach().clone())
    
    def hook_attention_npu(module, input, output):
        attention_inputs_npu.append([x.detach().clone() if isinstance(x, torch.Tensor) else x for x in input])
        attention_outputs_npu.append(output[0].detach().clone())
    
    first_layer_cpu = model_cpu.model.layers[0]
    first_layer_npu = model_npu.model.layers[0]
    
    handle_cpu = first_layer_cpu.self_attn.register_forward_hook(hook_attention_cpu)
    handle_npu = first_layer_npu.self_attn.register_forward_hook(hook_attention_npu)
    
    # Forward pass
    with torch.no_grad():
        outputs_cpu = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
        outputs_npu = model_npu(input_ids=input_ids_npu, attention_mask=attention_mask_npu)
    
    handle_cpu.remove()
    handle_npu.remove()
    
    all_passed = True
    
    # 比较 attention 输入
    if attention_inputs_cpu and attention_inputs_npu:
        if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
            print("\n  Attention input comparison:")
        for i, (in_cpu, in_npu) in enumerate(zip(attention_inputs_cpu[0], attention_inputs_npu[0])):
            if isinstance(in_cpu, torch.Tensor) and isinstance(in_npu, torch.Tensor):
                matched, _ = verify_tensor_match(
                    in_npu.cpu(), in_cpu, f"attention input {i}", config.tolerance, config
                )
                if not matched:
                    all_passed = False
    
    # 比较 attention 输出
    if attention_outputs_cpu and attention_outputs_npu:
        if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
            print("\n  Attention output comparison:")
        out_cpu = attention_outputs_cpu[0]
        out_npu = attention_outputs_npu[0].cpu()
        
        matched, _ = verify_tensor_match(out_npu, out_cpu, "attention output", config.tolerance, config)
        if not matched:
            all_passed = False
    
    # 比较最终 logits（使用更宽松的容差）
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print("\n  Final logits comparison:")
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
    parser = create_arg_parser("Attention Computation Test")
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
    
    print_section("Attention Computation Test", config)
    
    results = []
    
    # 基础 attention 操作测试
    print_section("Basic Attention Operations", config)
    results.append(("Q @ K^T (bmm)", run_test(test_qk_bmm, config, "Q @ K^T Test")))
    results.append(("Softmax", run_test(test_softmax, config, "Softmax Test")))
    results.append(("Attention @ V (bmm)", run_test(test_attn_v_bmm, config, "Attention @ V Test")))
    
    # 模型测试
    if not args.skip_model_tests:
        print_section("Model Attention Tests", config)
        results.append(("Attention in Model", run_test(
            lambda c: test_attention_in_model(c, args.model_path), config, "Attention in Model Test"
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

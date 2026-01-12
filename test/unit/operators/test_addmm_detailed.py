#!/usr/bin/env python
"""
详细测试 addmm 操作

测试内容:
1. 基本 addmm 操作
2. 使用实际模型权重测试 (gate_proj, up_proj, down_proj)

使用方法:
    python test/unit/operators/test_addmm_detailed.py
    python test/unit/operators/test_addmm_detailed.py -vv
    python test/unit/operators/test_addmm_detailed.py --tolerance 1e-4
    python test/unit/operators/test_addmm_detailed.py --model-path Qwen/Qwen2.5-0.5B
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


def test_basic_addmm(config: TestConfig) -> bool:
    """基本 addmm 测试"""
    print_step("Basic addmm test", config)
    
    device = torch.device(config.device)
    
    # addmm: output = beta * input + alpha * (mat1 @ mat2)
    # 在 PyTorch 中，linear layer 使用 addmm: output = input @ weight^T + bias
    batch_size = 1
    seq_len = 3
    hidden_size = 896
    out_features = 3584
    
    input_cpu = torch.randn(batch_size * seq_len, hidden_size, dtype=torch.float32)
    weight_cpu = torch.randn(out_features, hidden_size, dtype=torch.float32)
    bias_cpu = torch.randn(out_features, dtype=torch.float32)
    
    input_npu = input_cpu.to(device)
    weight_npu = weight_cpu.to(device)
    bias_npu = bias_cpu.to(device)
    
    # addmm(bias, input, weight^T) = input @ weight^T + bias
    result_cpu = torch.addmm(bias_cpu, input_cpu, weight_cpu.t())
    result_npu = torch.addmm(bias_npu, input_npu, weight_npu.t()).cpu()
    
    matched, info = verify_tensor_match(result_npu, result_cpu, "addmm output", config.tolerance, config)
    
    if not matched and config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  CPU result (first 5): {result_cpu[0, :5]}")
        print(f"  NPU result (first 5): {result_npu[0, :5]}")
    
    return matched


def test_ffn_projections(config: TestConfig, model_path: str) -> bool:
    """使用实际模型权重测试 FFN 投影层"""
    print_step("FFN projections with model weights", config)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ⚠ transformers not installed, skipping model tests")
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
    
    # 获取第一层 FFN 的权重
    first_layer = model_cpu.model.layers[0]
    ffn_gate_proj = first_layer.mlp.gate_proj
    ffn_up_proj = first_layer.mlp.up_proj
    ffn_down_proj = first_layer.mlp.down_proj
    
    # 创建测试输入 (假设 hidden_size=896)
    hidden_size = first_layer.mlp.gate_proj.weight.shape[1]
    test_input = torch.randn(1, 3, hidden_size, dtype=torch.float32)
    test_input_2d = test_input.view(-1, hidden_size)
    
    all_passed = True
    
    # 测试 gate_proj
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print("\n  Testing gate_proj...")
    gate_weight_cpu = ffn_gate_proj.weight.data
    gate_bias_cpu = ffn_gate_proj.bias.data if ffn_gate_proj.bias is not None else None
    
    gate_weight_npu = gate_weight_cpu.to(device)
    gate_bias_npu = gate_bias_cpu.to(device) if gate_bias_cpu is not None else None
    
    bias_tensor_cpu = gate_bias_cpu if gate_bias_cpu is not None else torch.zeros(gate_weight_cpu.size(0))
    bias_tensor_npu = gate_bias_npu if gate_bias_npu is not None else torch.zeros(gate_weight_npu.size(0), device=device)
    
    gate_out_cpu = torch.addmm(bias_tensor_cpu, test_input_2d, gate_weight_cpu.t())
    gate_out_npu = torch.addmm(bias_tensor_npu, test_input_2d.to(device), gate_weight_npu.t()).cpu()
    
    matched, _ = verify_tensor_match(gate_out_npu, gate_out_cpu, "gate_proj", config.tolerance, config)
    if not matched:
        all_passed = False
    
    # 测试 up_proj
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print("\n  Testing up_proj...")
    up_weight_cpu = ffn_up_proj.weight.data
    up_bias_cpu = ffn_up_proj.bias.data if ffn_up_proj.bias is not None else None
    
    up_weight_npu = up_weight_cpu.to(device)
    up_bias_npu = up_bias_cpu.to(device) if up_bias_cpu is not None else None
    
    bias_tensor_cpu = up_bias_cpu if up_bias_cpu is not None else torch.zeros(up_weight_cpu.size(0))
    bias_tensor_npu = up_bias_npu if up_bias_npu is not None else torch.zeros(up_weight_npu.size(0), device=device)
    
    up_out_cpu = torch.addmm(bias_tensor_cpu, test_input_2d, up_weight_cpu.t())
    up_out_npu = torch.addmm(bias_tensor_npu, test_input_2d.to(device), up_weight_npu.t()).cpu()
    
    matched, _ = verify_tensor_match(up_out_npu, up_out_cpu, "up_proj", config.tolerance, config)
    if not matched:
        all_passed = False
    
    # 测试 down_proj
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print("\n  Testing down_proj...")
    down_weight_cpu = ffn_down_proj.weight.data
    down_bias_cpu = ffn_down_proj.bias.data if ffn_down_proj.bias is not None else None
    
    down_weight_npu = down_weight_cpu.to(device)
    down_bias_npu = down_bias_cpu.to(device) if down_bias_cpu is not None else None
    
    # 先计算 gate 和 up 的输出（模拟 FFN 的前半部分）
    silu_gate = gate_out_cpu * torch.sigmoid(gate_out_cpu)
    ffn_intermediate = silu_gate * up_out_cpu
    
    bias_tensor_cpu = down_bias_cpu if down_bias_cpu is not None else torch.zeros(down_weight_cpu.size(0))
    bias_tensor_npu = down_bias_npu if down_bias_npu is not None else torch.zeros(down_weight_npu.size(0), device=device)
    
    down_out_cpu = torch.addmm(bias_tensor_cpu, ffn_intermediate, down_weight_cpu.t())
    
    # NPU 版本
    silu_gate_npu = gate_out_npu * torch.sigmoid(gate_out_npu)
    ffn_intermediate_npu = silu_gate_npu * up_out_npu
    down_out_npu = torch.addmm(bias_tensor_npu, ffn_intermediate_npu.to(device), down_weight_npu.t()).cpu()
    
    matched, _ = verify_tensor_match(down_out_npu, down_out_cpu, "down_proj", config.tolerance, config)
    if not matched:
        all_passed = False
    
    return all_passed


def main():
    parser = create_arg_parser("Detailed addmm Test")
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen2.5-0.5B',
        help='Model path for FFN tests (default: Qwen/Qwen2.5-0.5B)'
    )
    parser.add_argument(
        '--skip-model-tests',
        action='store_true',
        help='Skip tests that require loading models'
    )
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_echo_npu()
    
    print_section("Detailed addmm Test", config)
    
    results = []
    
    # 基本测试
    results.append(("Basic addmm", run_test(test_basic_addmm, config, "Basic addmm Test")))
    
    # 模型测试
    if not args.skip_model_tests:
        results.append(("FFN Projections", run_test(
            lambda c: test_ffn_projections(c, args.model_path), config, "FFN Projections Test"
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

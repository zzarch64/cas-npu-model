#!/usr/bin/env python
"""
算子精度测试

测试内容:
1. 基础算子测试 (mm, bmm, add, addmm)
2. 模型第一层输出对比
3. 逐步检查每个 transformer layer

使用方法:
    python test/unit/operators/test_operator_accuracy.py
    python test/unit/operators/test_operator_accuracy.py -vv
    python test/unit/operators/test_operator_accuracy.py --tolerance 1e-4
    python test/unit/operators/test_operator_accuracy.py --model-path Qwen/Qwen2.5-0.5B
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


def test_mm(config: TestConfig) -> bool:
    """测试 mm (矩阵乘法)"""
    print_step("mm (matrix multiplication)", config)
    
    device = torch.device(config.device)
    
    A_cpu = torch.randn(3, 4, dtype=torch.float32)
    B_cpu = torch.randn(4, 5, dtype=torch.float32)
    A_npu = A_cpu.to(device)
    B_npu = B_cpu.to(device)
    
    result_cpu = torch.mm(A_cpu, B_cpu)
    result_npu = torch.mm(A_npu, B_npu).cpu()
    
    matched, info = verify_tensor_match(result_npu, result_cpu, "mm output", config.tolerance, config)
    
    if not matched and config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  CPU result (first 3x3):\n{result_cpu[:3, :3]}")
        print(f"  NPU result (first 3x3):\n{result_npu[:3, :3]}")
    
    return matched


def test_bmm(config: TestConfig) -> bool:
    """测试 bmm (批量矩阵乘法)"""
    print_step("bmm (batch matrix multiplication)", config)
    
    device = torch.device(config.device)
    
    A_cpu = torch.randn(2, 3, 4, dtype=torch.float32)
    B_cpu = torch.randn(2, 4, 5, dtype=torch.float32)
    A_npu = A_cpu.to(device)
    B_npu = B_cpu.to(device)
    
    result_cpu = torch.bmm(A_cpu, B_cpu)
    result_npu = torch.bmm(A_npu, B_npu).cpu()
    
    matched, info = verify_tensor_match(result_npu, result_cpu, "bmm output", config.tolerance, config)
    
    if not matched and config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  CPU result (first batch, first 2x2):\n{result_cpu[0, :2, :2]}")
        print(f"  NPU result (first batch, first 2x2):\n{result_npu[0, :2, :2]}")
    
    return matched


def test_add(config: TestConfig) -> bool:
    """测试 add"""
    print_step("add", config)
    
    device = torch.device(config.device)
    
    A_cpu = torch.randn(3, 4, dtype=torch.float32)
    B_cpu = torch.randn(3, 4, dtype=torch.float32)
    A_npu = A_cpu.to(device)
    B_npu = B_cpu.to(device)
    
    result_cpu = torch.add(A_cpu, B_cpu)
    result_npu = torch.add(A_npu, B_npu).cpu()
    
    matched, info = verify_tensor_match(result_npu, result_cpu, "add output", config.tolerance, config)
    return matched


def test_addmm(config: TestConfig) -> bool:
    """测试 addmm"""
    print_step("addmm", config)
    
    device = torch.device(config.device)
    
    A_cpu = torch.randn(3, 5, dtype=torch.float32)
    B_cpu = torch.randn(3, 4, dtype=torch.float32)
    C_cpu = torch.randn(4, 5, dtype=torch.float32)
    A_npu = A_cpu.to(device)
    B_npu = B_cpu.to(device)
    C_npu = C_cpu.to(device)
    
    result_cpu = torch.addmm(A_cpu, B_cpu, C_cpu)
    result_npu = torch.addmm(A_npu, B_npu, C_npu).cpu()
    
    matched, info = verify_tensor_match(result_npu, result_cpu, "addmm output", config.tolerance, config)
    return matched


def test_model_first_layer(config: TestConfig, model_path: str) -> bool:
    """测试模型第一层输出对比"""
    print_step("Model first layer output comparison", config)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ⚠ transformers not installed, skipping model tests")
        return True  # 跳过但不算失败
    
    device = torch.device(config.device)
    
    # 加载模型
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"  Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    
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
    model_npu = model_npu.to(device)
    model_npu.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备输入
    prompt = "讲个笑话"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    # Hook 第一层的输出
    outputs_cpu_layer = []
    outputs_npu_layer = []
    
    def hook_cpu(module, input, output):
        outputs_cpu_layer.append(output[0].detach().clone())
    
    def hook_npu(module, input, output):
        outputs_npu_layer.append(output[0].detach().clone())
    
    first_layer_cpu = model_cpu.model.layers[0]
    first_layer_npu = model_npu.model.layers[0]
    
    handle_cpu = first_layer_cpu.register_forward_hook(hook_cpu)
    handle_npu = first_layer_npu.register_forward_hook(hook_npu)
    
    # Forward pass
    input_ids_npu = input_ids.to(device)
    attention_mask_npu = attention_mask.to(device) if attention_mask is not None else None
    
    with torch.no_grad():
        outputs_cpu = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
        outputs_npu = model_npu(input_ids=input_ids_npu, attention_mask=attention_mask_npu)
    
    handle_cpu.remove()
    handle_npu.remove()
    
    # 比较第一层输出
    if outputs_cpu_layer and outputs_npu_layer:
        layer_out_cpu = outputs_cpu_layer[0]
        layer_out_npu = outputs_npu_layer[0].cpu()
        
        matched, info = verify_tensor_match(
            layer_out_npu, layer_out_cpu, "first layer output", config.tolerance, config
        )
        return matched
    
    return False


def test_transformer_layers(config: TestConfig, model_path: str, num_layers: int = 3) -> bool:
    """逐步检查每个 transformer layer（使用 hooks 捕获完整模型推理时的输出）"""
    print_step(f"Check first {num_layers} transformer layers", config)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print("  ⚠ transformers not installed, skipping model tests")
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
    
    all_passed = True
    total_layers = len(model_cpu.model.layers)
    
    # 用 hooks 捕获每层输出（通过完整模型推理）
    layer_outputs_cpu = {}
    layer_outputs_npu = {}
    
    def make_hook_cpu(layer_idx):
        def hook(module, input, output):
            layer_outputs_cpu[layer_idx] = output[0].detach().clone()
        return hook
    
    def make_hook_npu(layer_idx):
        def hook(module, input, output):
            layer_outputs_npu[layer_idx] = output[0].detach().clone()
        return hook
    
    # 注册 hooks
    handles = []
    for i in range(min(num_layers, total_layers)):
        handles.append(model_cpu.model.layers[i].register_forward_hook(make_hook_cpu(i)))
        handles.append(model_npu.model.layers[i].register_forward_hook(make_hook_npu(i)))
    
    # 完整模型推理（这样 position_embeddings 和 attention_mask 会被正确处理）
    with torch.no_grad():
        _ = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
        _ = model_npu(input_ids=input_ids_npu, attention_mask=attention_mask_npu)
    
    # 清理 hooks
    for h in handles:
        h.remove()
    
    # 比较每层输出
    for i in range(min(num_layers, total_layers)):
        if i in layer_outputs_cpu and i in layer_outputs_npu:
            out_cpu = layer_outputs_cpu[i]
            out_npu = layer_outputs_npu[i].cpu()
            
            diff = (out_cpu - out_npu).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            # 计算相对差异
            out_abs = out_cpu.abs()
            relative_diff = (diff / (out_abs + 1e-8)).max().item()
            
            # 对于 transformer layer，误差会随层数累积
            # 使用更宽松的容差：基础 1e-3，或相对差异 < 1%
            layer_tolerance = max(config.tolerance, 1e-3)
            is_ok = max_diff <= layer_tolerance or relative_diff < 0.01
            
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                status = "✓" if is_ok else "✗"
                print(f"    {status} Layer {i}: max_diff={max_diff:.6f}, relative={relative_diff:.4%}")
            
            if not is_ok:
                all_passed = False
    
    return all_passed


def main():
    parser = create_arg_parser("Operator Accuracy Test")
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen2.5-0.5B',
        help='Model path for transformer tests (default: Qwen/Qwen2.5-0.5B)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        help='Number of transformer layers to test (default: 3)'
    )
    parser.add_argument(
        '--skip-model-tests',
        action='store_true',
        help='Skip tests that require loading models'
    )
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_cas_npu()
    
    print_section("Operator Accuracy Test", config)
    
    results = []
    
    # 基础算子测试
    print_section("Basic Operator Tests", config)
    results.append(("mm", run_test(test_mm, config, "mm Test")))
    results.append(("bmm", run_test(test_bmm, config, "bmm Test")))
    results.append(("add", run_test(test_add, config, "add Test")))
    results.append(("addmm", run_test(test_addmm, config, "addmm Test")))
    
    # 模型测试
    if not args.skip_model_tests:
        print_section("Model Tests", config)
        results.append(("Model First Layer", run_test(
            lambda c: test_model_first_layer(c, args.model_path), config, "Model First Layer Test"
        )))
        results.append(("Transformer Layers", run_test(
            lambda c: test_transformer_layers(c, args.model_path, args.num_layers), config, "Transformer Layers Test"
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

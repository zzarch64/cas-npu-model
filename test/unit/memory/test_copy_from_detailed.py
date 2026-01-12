#!/usr/bin/env python
"""
详细测试数据拷贝操作 (_copy_from)

测试内容:
1. 基本拷贝测试 (CPU->NPU, NPU->CPU, NPU->NPU)
2. 非 contiguous tensor 拷贝 (transpose, slice, view)
3. 3D tensor 拷贝
4. 模型数据传递测试

使用方法:
    python test/unit/memory/test_copy_from_detailed.py
    python test/unit/memory/test_copy_from_detailed.py -vv
    python test/unit/memory/test_copy_from_detailed.py --model-path Qwen/Qwen2.5-0.5B
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


def test_cpu_to_npu(config: TestConfig) -> bool:
    """测试 CPU -> NPU 拷贝"""
    print_step("CPU -> NPU -> CPU copy", config)
    
    device = torch.device(config.device)
    
    x_cpu = torch.randn(3, 4, dtype=torch.float32)
    x_npu = x_cpu.to(device)
    x_back = x_npu.cpu()
    
    matched, info = verify_tensor_match(x_back, x_cpu, "CPU->NPU->CPU", 1e-7, config)
    return matched


def test_npu_to_cpu(config: TestConfig) -> bool:
    """测试 NPU -> CPU 拷贝"""
    print_step("NPU -> CPU -> NPU copy", config)
    
    device = torch.device(config.device)
    
    x_npu = torch.randn(3, 4, dtype=torch.float32, device=device)
    x_cpu = x_npu.cpu()
    x_back = x_cpu.to(device).cpu()
    
    matched, info = verify_tensor_match(x_back, x_cpu, "NPU->CPU->NPU", 1e-7, config)
    return matched


def test_npu_to_npu(config: TestConfig) -> bool:
    """测试 NPU -> NPU 克隆"""
    print_step("NPU -> NPU clone", config)
    
    device = torch.device(config.device)
    
    x_npu = torch.randn(3, 4, dtype=torch.float32, device=device)
    y_npu = x_npu.clone()
    
    matched, info = verify_tensor_match(y_npu.cpu(), x_npu.cpu(), "NPU->NPU clone", 1e-7, config)
    return matched


def test_transpose_copy(config: TestConfig) -> bool:
    """测试 transpose (非 contiguous) 拷贝"""
    print_step("Transpose tensor copy", config)
    
    device = torch.device(config.device)
    
    x_cpu = torch.randn(3, 4, dtype=torch.float32)
    x_t_cpu = x_cpu.t()  # 非 contiguous
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Original is contiguous: {x_cpu.is_contiguous()}")
        print(f"  Transposed is contiguous: {x_t_cpu.is_contiguous()}")
    
    x_t_npu = x_t_cpu.to(device)
    x_t_back = x_t_npu.cpu()
    
    matched, info = verify_tensor_match(x_t_back, x_t_cpu, "transpose copy", 1e-7, config)
    return matched


def test_slice_copy(config: TestConfig) -> bool:
    """测试 slice (非 contiguous) 拷贝"""
    print_step("Sliced tensor copy", config)
    
    device = torch.device(config.device)
    
    x_cpu = torch.randn(10, 10, dtype=torch.float32)
    x_slice_cpu = x_cpu[::2, ::2]  # 非 contiguous
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Sliced is contiguous: {x_slice_cpu.is_contiguous()}")
    
    x_slice_npu = x_slice_cpu.to(device)
    x_slice_back = x_slice_npu.cpu()
    
    matched, info = verify_tensor_match(x_slice_back, x_slice_cpu, "slice copy", 1e-7, config)
    return matched


def test_view_copy(config: TestConfig) -> bool:
    """测试 view/reshape 拷贝"""
    print_step("Reshaped tensor copy", config)
    
    device = torch.device(config.device)
    
    x_cpu = torch.randn(12, dtype=torch.float32)
    x_view_cpu = x_cpu.view(3, 4)
    
    x_view_npu = x_view_cpu.to(device)
    x_view_back = x_view_npu.cpu()
    
    matched, info = verify_tensor_match(x_view_back, x_view_cpu, "view copy", 1e-7, config)
    return matched


def test_3d_contiguous(config: TestConfig) -> bool:
    """测试 3D contiguous tensor 拷贝"""
    print_step("3D contiguous tensor copy", config)
    
    device = torch.device(config.device)
    
    x_cpu = torch.randn(1, 3, 896, dtype=torch.float32)
    x_npu = x_cpu.to(device)
    x_back = x_npu.cpu()
    
    matched, info = verify_tensor_match(x_back, x_cpu, "3D contiguous copy", 1e-7, config)
    return matched


def test_3d_permuted(config: TestConfig) -> bool:
    """测试 3D permuted (非 contiguous) tensor 拷贝"""
    print_step("3D permuted tensor copy", config)
    
    device = torch.device(config.device)
    
    x_cpu = torch.randn(1, 896, 3, dtype=torch.float32)
    x_perm_cpu = x_cpu.permute(0, 2, 1)  # [1, 3, 896] 但非 contiguous
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  Permuted is contiguous: {x_perm_cpu.is_contiguous()}")
        print(f"  Permuted stride: {x_perm_cpu.stride()}")
    
    x_perm_npu = x_perm_cpu.to(device)
    x_perm_back = x_perm_npu.cpu()
    
    matched, info = verify_tensor_match(x_perm_back, x_perm_cpu, "3D permuted copy", 1e-7, config)
    return matched


def test_model_data_transfer(config: TestConfig, model_path: str) -> bool:
    """测试模型数据传递"""
    print_step("Model data transfer test", config)
    
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
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "讲个笑话"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    input_ids_npu = input_ids.to(device)
    attention_mask_npu = attention_mask.to(device) if attention_mask is not None else None
    
    # 获取 attention 输出
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
    
    all_passed = True
    
    if attn_out_npu is not None:
        # 测试 NPU 输出拷贝回 CPU 并再次拷贝
        attn_out_npu_to_cpu = attn_out_npu.cpu()
        attn_out_back_to_npu = attn_out_npu_to_cpu.to(device)
        attn_out_back_to_cpu = attn_out_back_to_npu.cpu()
        
        matched, _ = verify_tensor_match(
            attn_out_back_to_cpu, attn_out_npu_to_cpu, "attention output round-trip", 1e-7, config
        )
        if not matched:
            all_passed = False
        
        # 测试 layer norm 权重拷贝
        ln_weight_cpu = first_layer_cpu.post_attention_layernorm.weight.data
        ln_weight_npu = first_layer_npu.post_attention_layernorm.weight.data.cpu()
        
        matched, _ = verify_tensor_match(ln_weight_npu, ln_weight_cpu, "layer norm weight", 1e-7, config)
        if not matched:
            all_passed = False
    
    return all_passed


def main():
    parser = create_arg_parser("Copy From Detailed Test")
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen2.5-0.5B',
        help='Model path for data transfer tests (default: Qwen/Qwen2.5-0.5B)'
    )
    parser.add_argument(
        '--skip-model-tests',
        action='store_true',
        help='Skip tests that require loading models'
    )
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_cas_npu()
    
    print_section("Copy From Detailed Test", config)
    
    results = []
    
    # 基本拷贝测试
    print_section("Basic Copy Tests", config)
    results.append(("CPU->NPU->CPU", run_test(test_cpu_to_npu, config, "CPU->NPU->CPU Test")))
    results.append(("NPU->CPU->NPU", run_test(test_npu_to_cpu, config, "NPU->CPU->NPU Test")))
    results.append(("NPU->NPU clone", run_test(test_npu_to_npu, config, "NPU->NPU Clone Test")))
    
    # 非 contiguous 测试
    print_section("Non-contiguous Copy Tests", config)
    results.append(("Transpose", run_test(test_transpose_copy, config, "Transpose Copy Test")))
    results.append(("Slice", run_test(test_slice_copy, config, "Slice Copy Test")))
    results.append(("View", run_test(test_view_copy, config, "View Copy Test")))
    
    # 3D tensor 测试
    print_section("3D Tensor Copy Tests", config)
    results.append(("3D Contiguous", run_test(test_3d_contiguous, config, "3D Contiguous Test")))
    results.append(("3D Permuted", run_test(test_3d_permuted, config, "3D Permuted Test")))
    
    # 模型数据传递测试
    if not args.skip_model_tests:
        print_section("Model Data Transfer Tests", config)
        results.append(("Model Data Transfer", run_test(
            lambda c: test_model_data_transfer(c, args.model_path), config, "Model Data Transfer Test"
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

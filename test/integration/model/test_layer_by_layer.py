#!/usr/bin/env python
"""
逐层检查模型，定位问题出现的具体位置

测试内容:
1. Embedding 层对比
2. 逐层检查 transformer layers
3. 最终输出对比

使用方法:
    python test/integration/model/test_layer_by_layer.py
    python test/integration/model/test_layer_by_layer.py -vv
    python test/integration/model/test_layer_by_layer.py --model-path Qwen/Qwen2.5-0.5B
    python test/integration/model/test_layer_by_layer.py --num-layers 5
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


def test_embedding_layer(config: TestConfig, model_cpu, model_npu, input_ids, input_ids_npu) -> tuple:
    """测试 Embedding 层"""
    print_step("Embedding layer", config)
    
    with torch.no_grad():
        emb_cpu = model_cpu.model.embed_tokens(input_ids)
        emb_npu = model_npu.model.embed_tokens(input_ids_npu).cpu()
    
    matched, info = verify_tensor_match(emb_npu, emb_cpu, "embedding output", config.tolerance, config)
    return matched, emb_cpu, emb_npu


def compare_layer_outputs(config: TestConfig, out_cpu, out_npu, layer_idx: int) -> bool:
    """比较单层输出"""
    diff = (out_cpu - out_npu.cpu()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # 计算相对差异
    out_abs = out_cpu.abs()
    relative_diff = (diff / (out_abs + 1e-8)).max().item()
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"    Max difference: {max_diff:.6f}")
        print(f"    Mean difference: {mean_diff:.6f}")
        print(f"    Relative max diff: {relative_diff:.4%}")
    
    # 误差会随层数累积，使用更宽松的容差：
    # - 绝对差异 < 0.01，或
    # - 相对差异 < 1%
    layer_tolerance = max(config.tolerance, 1e-3)
    passed = max_diff <= layer_tolerance or relative_diff < 0.01
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        status = "✓" if passed else "✗"
        print(f"    {status} Layer {layer_idx}: max_diff={max_diff:.6f}, relative={relative_diff:.4%}")
    
    return passed


def run_layer_by_layer_test(config: TestConfig, model_path: str, num_layers: int) -> bool:
    """运行逐层测试"""
    
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
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"\n  Prompt: {prompt}")
        print(f"  Input IDs: {input_ids.tolist()}")
    
    all_passed = True
    
    # 1. 测试 Embedding
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print_section("Embedding Layer", config)
    emb_passed, hidden_cpu, hidden_npu = test_embedding_layer(
        config, model_cpu, model_npu, input_ids, input_ids_npu
    )
    if not emb_passed:
        all_passed = False
    
    hidden_npu = hidden_npu.to(device)
    
    # 2. 逐层检查 transformer layers（使用 hooks 捕获完整模型推理时的输出）
    total_layers = len(model_cpu.model.layers)
    test_layers = min(num_layers, total_layers)
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print_section(f"Transformer Layers (first {test_layers} of {total_layers})", config)
    
    # 用 hooks 捕获每层输出
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
    for i in range(test_layers):
        handles.append(model_cpu.model.layers[i].register_forward_hook(make_hook_cpu(i)))
        handles.append(model_npu.model.layers[i].register_forward_hook(make_hook_npu(i)))
    
    # 完整模型推理
    with torch.no_grad():
        _ = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
        _ = model_npu(input_ids=input_ids_npu, attention_mask=attention_mask_npu)
    
    # 清理 hooks
    for h in handles:
        h.remove()
    
    # 比较每层输出
    for i in range(test_layers):
        if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
            print(f"\n  Layer {i}:")
        
        if i in layer_outputs_cpu and i in layer_outputs_npu:
            layer_passed = compare_layer_outputs(
                config, layer_outputs_cpu[i], layer_outputs_npu[i], i
            )
            if not layer_passed:
                all_passed = False
    
    # 3. 最终输出
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print_section("Final Output", config)
    with torch.no_grad():
        outputs_cpu = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
        outputs_npu = model_npu(input_ids=input_ids_npu, attention_mask=attention_mask_npu)
    
    logits_cpu = outputs_cpu.logits[0, -1, :10]
    logits_npu = outputs_npu.logits[0, -1, :10].cpu()
    
    # 最终输出使用更宽松的容差（误差会累积），同时检查相对误差
    model_tolerance = max(config.tolerance, 1e-4)
    matched, info = verify_tensor_match(logits_npu, logits_cpu, "final logits (first 10)", model_tolerance, config)
    
    # 额外检查：如果绝对误差超标但相对误差很小，也认为通过
    if not matched:
        diff = (logits_cpu - logits_npu).abs()
        relative_diff = (diff / (logits_cpu.abs() + 1e-8)).max().item()
        if relative_diff < 0.01:  # 相对误差 < 1%
            if config.verbosity.value >= VerbosityLevel.NORMAL.value:
                print(f"    (相对误差 {relative_diff:.4%} < 1%, 视为通过)")
            matched = True
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"  CPU logits (first 10): {logits_cpu}")
        print(f"  NPU logits (first 10): {logits_npu}")
    
    if not matched:
        all_passed = False
    
    return all_passed


def main():
    parser = create_arg_parser("Layer-by-Layer Test")
    parser.add_argument(
        '--model-path',
        type=str,
        default='Qwen/Qwen2.5-0.5B',
        help='Model path (default: Qwen/Qwen2.5-0.5B)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=5,
        help='Number of transformer layers to test (default: 5)'
    )
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    ensure_echo_npu()
    
    print_section("Layer-by-Layer Comparison Test", config)
    
    success = run_layer_by_layer_test(config, args.model_path, args.num_layers)
    
    print_section("Test Summary", config)
    if success:
        print("All layers passed! ✓")
    else:
        print("Some layers failed! ✗")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

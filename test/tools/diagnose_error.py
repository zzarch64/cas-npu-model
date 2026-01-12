#!/usr/bin/env python
"""
诊断误差来源：区分计算错误和精度问题

判断依据：
1. 误差分布 - 计算错误通常有系统性偏差，精度问题是随机分布
2. 误差增长 - 精度问题线性增长，计算错误可能指数增长
3. 相关性分析 - 检查误差是否与输入值相关
4. 特定位置分析 - 查看最大误差出现的位置是否有规律
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_framework import ensure_echo_npu, TestConfig, create_arg_parser
import torch
import numpy as np


def analyze_error(cpu_tensor, npu_tensor, name="tensor"):
    """详细分析误差特征"""
    diff = (cpu_tensor - npu_tensor).abs()
    signed_diff = cpu_tensor - npu_tensor
    
    print(f"\n{'='*60}")
    print(f"误差分析: {name}")
    print(f"{'='*60}")
    
    # 基本统计
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()
    median_diff = diff.median().item()
    
    print(f"\n[1] 基本误差统计:")
    print(f"    最大绝对误差: {max_diff:.6e}")
    print(f"    平均绝对误差: {mean_diff:.6e}")
    print(f"    误差标准差:   {std_diff:.6e}")
    print(f"    误差中位数:   {median_diff:.6e}")
    
    # 相对误差
    cpu_abs = cpu_tensor.abs()
    relative_diff = diff / (cpu_abs + 1e-8)
    max_rel = relative_diff.max().item()
    mean_rel = relative_diff.mean().item()
    
    print(f"\n[2] 相对误差:")
    print(f"    最大相对误差: {max_rel:.4%}")
    print(f"    平均相对误差: {mean_rel:.4%}")
    
    # 系统性偏差分析（如果有计算错误，可能会有系统性偏差）
    bias = signed_diff.mean().item()
    bias_std = signed_diff.std().item()
    
    print(f"\n[3] 系统性偏差分析 (判断计算错误):")
    print(f"    偏差均值: {bias:.6e}")
    print(f"    偏差标准差: {bias_std:.6e}")
    print(f"    偏差/标准差比: {abs(bias)/bias_std if bias_std > 0 else 0:.4f}")
    
    # 判断：如果 |bias| / std > 0.1，可能有系统性偏差
    if abs(bias) / bias_std > 0.1 if bias_std > 0 else False:
        print(f"    ⚠ 检测到系统性偏差！可能存在计算错误")
    else:
        print(f"    ✓ 无明显系统性偏差，误差呈随机分布（符合精度问题特征）")
    
    # 误差分布分析
    zero_count = (diff < 1e-7).sum().item()
    small_count = (diff < 1e-5).sum().item()
    medium_count = (diff < 1e-3).sum().item()
    total = diff.numel()
    
    print(f"\n[4] 误差分布:")
    print(f"    误差 < 1e-7: {zero_count}/{total} ({100*zero_count/total:.2f}%)")
    print(f"    误差 < 1e-5: {small_count}/{total} ({100*small_count/total:.2f}%)")
    print(f"    误差 < 1e-3: {medium_count}/{total} ({100*medium_count/total:.2f}%)")
    
    # 最大误差位置分析
    max_idx = diff.argmax()
    if diff.dim() == 3:
        idx = torch.unravel_index(max_idx, diff.shape)
        print(f"\n[5] 最大误差位置:")
        print(f"    位置: [{idx[0].item()}, {idx[1].item()}, {idx[2].item()}]")
        print(f"    CPU值: {cpu_tensor[idx].item():.6f}")
        print(f"    NPU值: {npu_tensor[idx].item():.6f}")
        print(f"    差异: {signed_diff[idx].item():.6e}")
    
    # 相关性分析：误差是否与输入值大小相关
    cpu_flat = cpu_tensor.flatten()
    diff_flat = diff.flatten()
    
    # 简单相关系数
    cpu_centered = cpu_flat - cpu_flat.mean()
    diff_centered = diff_flat - diff_flat.mean()
    correlation = (cpu_centered * diff_centered).sum() / (cpu_centered.norm() * diff_centered.norm() + 1e-8)
    
    print(f"\n[6] 误差-数值相关性:")
    print(f"    相关系数: {correlation.item():.4f}")
    if abs(correlation.item()) > 0.3:
        print(f"    ⚠ 误差与数值大小有相关性，可能是溢出/下溢问题")
    else:
        print(f"    ✓ 误差与数值大小无明显相关，符合精度问题特征")
    
    # 综合判断
    print(f"\n{'='*60}")
    print(f"综合诊断:")
    print(f"{'='*60}")
    
    issues = []
    if abs(bias) / bias_std > 0.1 if bias_std > 0 else False:
        issues.append("系统性偏差")
    if abs(correlation.item()) > 0.3:
        issues.append("数值相关误差")
    if max_rel > 0.05:  # 5%
        issues.append("相对误差较大")
    
    if issues:
        print(f"  ⚠ 可能存在计算问题: {', '.join(issues)}")
        print(f"  建议: 检查相关算子实现")
    else:
        print(f"  ✓ 误差特征符合浮点精度问题:")
        print(f"    - 误差随机分布，无系统性偏差")
        print(f"    - 误差与数值大小无关")
        print(f"    - 相对误差在可接受范围内")
        print(f"  结论: 这是正常的浮点运算精度差异，不是计算错误")
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'bias': bias,
        'max_rel': max_rel,
        'correlation': correlation.item(),
        'is_precision_issue': len(issues) == 0
    }


def diagnose_model(model_path: str, num_layers: int = 5):
    """诊断模型各层误差"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers not installed")
        return
    
    device = torch.device("echo_npu:0")
    
    print(f"加载模型: {model_path}")
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
    
    # 收集每层输出
    layer_outputs_cpu = {}
    layer_outputs_npu = {}
    
    def make_hook_cpu(idx):
        def hook(m, i, o):
            layer_outputs_cpu[idx] = o[0].detach().clone()
        return hook
    
    def make_hook_npu(idx):
        def hook(m, i, o):
            layer_outputs_npu[idx] = o[0].detach().clone()
        return hook
    
    handles = []
    total_layers = len(model_cpu.model.layers)
    for i in range(min(num_layers, total_layers)):
        handles.append(model_cpu.model.layers[i].register_forward_hook(make_hook_cpu(i)))
        handles.append(model_npu.model.layers[i].register_forward_hook(make_hook_npu(i)))
    
    # 推理
    with torch.no_grad():
        out_cpu = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
        out_npu = model_npu(input_ids=input_ids_npu, attention_mask=attention_mask_npu)
    
    for h in handles:
        h.remove()
    
    # 分析每层
    print("\n" + "="*80)
    print("逐层误差诊断")
    print("="*80)
    
    layer_results = []
    for i in range(min(num_layers, total_layers)):
        result = analyze_error(
            layer_outputs_cpu[i], 
            layer_outputs_npu[i].cpu(), 
            f"Layer {i}"
        )
        layer_results.append(result)
    
    # 分析误差增长趋势
    print("\n" + "="*80)
    print("误差增长趋势分析")
    print("="*80)
    
    max_diffs = [r['max_diff'] for r in layer_results]
    print(f"\n各层最大误差: {[f'{d:.6e}' for d in max_diffs]}")
    
    if len(max_diffs) >= 2:
        # 计算增长率
        growth_rates = [max_diffs[i+1] / max_diffs[i] if max_diffs[i] > 0 else 0 
                       for i in range(len(max_diffs)-1)]
        avg_growth = np.mean(growth_rates)
        
        print(f"增长率: {[f'{r:.2f}x' for r in growth_rates]}")
        print(f"平均增长率: {avg_growth:.2f}x")
        
        if avg_growth > 10:
            print(f"⚠ 误差指数增长，可能存在计算错误!")
        elif avg_growth > 2:
            print(f"⚠ 误差增长较快，建议检查")
        else:
            print(f"✓ 误差增长在正常范围内（符合精度累积特征）")
    
    # 最终输出分析
    print("\n")
    logits_cpu = out_cpu.logits[0, -1, :]
    logits_npu = out_npu.logits[0, -1, :].cpu()
    analyze_error(logits_cpu, logits_npu, "Final Logits")
    
    # 检查 top-k 预测是否一致
    print("\n" + "="*80)
    print("Top-5 预测对比 (最重要的判断)")
    print("="*80)
    
    top5_cpu = logits_cpu.topk(5)
    top5_npu = logits_npu.topk(5)
    
    print(f"\nCPU Top-5 tokens: {top5_cpu.indices.tolist()}")
    print(f"NPU Top-5 tokens: {top5_npu.indices.tolist()}")
    
    if top5_cpu.indices.tolist() == top5_npu.indices.tolist():
        print(f"✓ Top-5 预测完全一致！")
        print(f"结论: 虽然数值有微小差异，但不影响最终预测结果，是精度问题而非计算错误")
    else:
        print(f"⚠ Top-5 预测不一致，可能需要检查")
        # 检查 top-1 是否一致
        if top5_cpu.indices[0] == top5_npu.indices[0]:
            print(f"但 Top-1 预测一致: {top5_cpu.indices[0].item()}")


def main():
    parser = create_arg_parser("Error Diagnosis Tool")
    parser.add_argument('--model-path', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--num-layers', type=int, default=5)
    args = parser.parse_args()
    
    ensure_echo_npu()
    
    print("="*80)
    print("误差诊断工具：区分计算错误 vs 精度问题")
    print("="*80)
    
    diagnose_model(args.model_path, args.num_layers)


if __name__ == "__main__":
    main()

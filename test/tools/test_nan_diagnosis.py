#!/usr/bin/env python
"""
NaN诊断工具 - 检查推理和训练过程中NaN的来源
"""

import sys
import os
import torch
import torch.nn as nn

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cas_npu
    print("✓ CAS-NPU extension imported successfully")
except ImportError as e:
    print(f"✗ Failed to import CAS-NPU extension: {e}")
    sys.exit(1)

def print_table(headers, rows, title=None):
    """打印表格"""
    if title:
        print(f"\n{title}")
        print("=" * 80)
    
    if not rows:
        return
    
    # 计算每列的最大宽度
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # 打印表头
    header_row = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))
    
    # 打印数据行
    for row in rows:
        data_row = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(data_row)

def check_tensor_for_nan(tensor, name="tensor", verbose=True):
    """检查tensor是否包含NaN或Inf"""
    if tensor is None:
        return False
    
    cpu_tensor = tensor.cpu() if tensor.device.type == 'cas_npu' else tensor
    
    has_nan = torch.isnan(cpu_tensor).any().item()
    has_inf = torch.isinf(cpu_tensor).any().item()
    
    if has_nan or has_inf:
        if verbose:
            print(f"  ⚠ {name} contains NaN/Inf!")
            print(f"    Shape: {cpu_tensor.shape}, Device: {tensor.device}")
            if has_nan:
                nan_count = torch.isnan(cpu_tensor).sum().item()
                print(f"    NaN count: {nan_count}/{cpu_tensor.numel()} ({nan_count/cpu_tensor.numel()*100:.2f}%)")
                # 找到第一个NaN的位置
                nan_indices = torch.nonzero(torch.isnan(cpu_tensor), as_tuple=False)
                if len(nan_indices) > 0:
                    print(f"    First NaN at: {nan_indices[0].tolist()}")
                    # 显示NaN周围的数值（如果可能）
                    if cpu_tensor.dim() <= 2 and cpu_tensor.numel() < 10000:
                        print(f"    Tensor values (first 20): {cpu_tensor.flatten()[:20]}")
            if has_inf:
                inf_count = torch.isinf(cpu_tensor).sum().item()
                print(f"    Inf count: {inf_count}/{cpu_tensor.numel()} ({inf_count/cpu_tensor.numel()*100:.2f}%)")
            # 显示统计信息
            finite_mask = torch.isfinite(cpu_tensor)
            if finite_mask.any():
                finite_values = cpu_tensor[finite_mask]
                print(f"    Finite values: min={finite_values.min().item():.6f}, max={finite_values.max().item():.6f}, mean={finite_values.mean().item():.6f}")
        return True
    
    return False

def test_memory_initialization():
    """测试内存初始化"""
    device = torch.device('cas_npu:0')
    
    # 创建多个tensor，检查是否包含NaN
    rows = []
    all_ok = True
    for i in range(5):
        t = torch.empty(100, 100, device=device)
        has_nan = check_tensor_for_nan(t, f"tensor_{i}", verbose=False)
        status = "✓ OK" if not has_nan else "✗ FAIL"
        if has_nan:
            all_ok = False
        rows.append([f"tensor_{i}", "(100, 100)", status])
    
    print_table(
        ["Tensor", "Shape", "Status"],
        rows,
        "Test 1: Memory Initialization Check"
    )
    return all_ok

def test_basic_operations():
    """测试基础操作是否产生NaN"""
    device = torch.device('cas_npu:0')
    rows = []
    all_ok = True
    
    # 测试加法
    a = torch.randn(10, 10, device=device)
    b = torch.randn(10, 10, device=device)
    c = a + b
    has_nan = check_tensor_for_nan(c, "add result", verbose=False)
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["add", "(10,10) + (10,10)", status])
    
    # 测试矩阵乘法
    a = torch.randn(10, 20, device=device)
    b = torch.randn(20, 15, device=device)
    c = torch.mm(a, b)
    has_nan = check_tensor_for_nan(c, "mm result", verbose=False)
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["mm", "(10,20) @ (20,15)", status])
    
    # 测试批量矩阵乘法
    a = torch.randn(2, 10, 20, device=device)
    b = torch.randn(2, 20, 15, device=device)
    c = torch.bmm(a, b)
    has_nan = check_tensor_for_nan(c, "bmm result", verbose=False)
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["bmm", "(2,10,20) @ (2,20,15)", status])
    
    print_table(
        ["Operation", "Shape", "Status"],
        rows,
        "Test 2: Basic Operations Check"
    )
    return all_ok

def test_linear_layer():
    """测试线性层"""
    device = torch.device('cas_npu:0')
    
    linear = nn.Linear(768, 3072).to(device)
    x = torch.randn(2, 10, 768, device=device)
    
    rows = []
    all_ok = True
    debug_info = []
    
    # 检查输入
    has_nan = check_tensor_for_nan(x, "input", verbose=False)
    if has_nan:
        debug_info.append(f"  ✗ Input contains NaN/Inf")
        debug_info.append(f"    Input stats: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["Input", "(2, 10, 768)", status])
    
    # 检查权重
    has_nan = check_tensor_for_nan(linear.weight, "weight", verbose=False)
    if has_nan:
        debug_info.append(f"  ✗ Weight contains NaN/Inf")
        debug_info.append(f"    Weight stats: min={linear.weight.min().item():.6f}, max={linear.weight.max().item():.6f}, mean={linear.weight.mean().item():.6f}")
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["Weight", "(3072, 768)", status])
    
    # 检查偏置
    has_nan = check_tensor_for_nan(linear.bias, "bias", verbose=False)
    if has_nan:
        debug_info.append(f"  ✗ Bias contains NaN/Inf")
        debug_info.append(f"    Bias stats: min={linear.bias.min().item():.6f}, max={linear.bias.max().item():.6f}, mean={linear.bias.mean().item():.6f}")
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["Bias", "(3072,)", status])
    
    # 前向传播
    try:
        y = linear(x)
        has_nan = check_tensor_for_nan(y, "linear output", verbose=False)
        if has_nan:
            debug_info.append(f"  ✗ Forward output contains NaN/Inf")
            debug_info.append(f"    Output stats: min={y.min().item():.6f}, max={y.max().item():.6f}, mean={y.mean().item():.6f}")
            # 检查是否是mm操作的问题
            debug_info.append(f"    Checking intermediate computation...")
            # 手动检查矩阵乘法
            x_flat = x.view(-1, 768)  # (20, 768)
            weight_t = linear.weight.t()  # (768, 3072)
            try:
                mm_result = torch.mm(x_flat, weight_t)  # (20, 3072)
                mm_has_nan = torch.isnan(mm_result).any() or torch.isinf(mm_result).any()
                if mm_has_nan:
                    debug_info.append(f"    ✗ mm operation produces NaN/Inf")
                    debug_info.append(f"      mm result stats: min={mm_result.min().item():.6f}, max={mm_result.max().item():.6f}")
                else:
                    debug_info.append(f"    ✓ mm operation OK")
                    # 检查bias加法
                    mm_plus_bias = mm_result + linear.bias.unsqueeze(0)
                    bias_has_nan = torch.isnan(mm_plus_bias).any() or torch.isinf(mm_plus_bias).any()
                    if bias_has_nan:
                        debug_info.append(f"    ✗ Adding bias produces NaN/Inf")
                    else:
                        debug_info.append(f"    ✓ Adding bias OK")
            except Exception as e:
                debug_info.append(f"    ✗ Error checking intermediate computation: {e}")
        status = "✓ OK" if not has_nan else "✗ FAIL"
        if has_nan:
            all_ok = False
        rows.append(["Forward Output", "(2, 10, 3072)", status])
    except Exception as e:
        debug_info.append(f"  ✗ Forward pass failed with exception: {e}")
        rows.append(["Forward Output", "ERROR", "✗ FAIL"])
        all_ok = False
        y = None
    
    # 反向传播
    if y is not None:
        try:
            loss = y.sum()
            loss.backward()
            
            has_nan = check_tensor_for_nan(linear.weight.grad, "weight grad", verbose=False)
            if has_nan:
                debug_info.append(f"  ✗ Weight gradient contains NaN/Inf")
                if linear.weight.grad is not None:
                    debug_info.append(f"    Weight grad stats: min={linear.weight.grad.min().item():.6f}, max={linear.weight.grad.max().item():.6f}")
            status = "✓ OK" if not has_nan else "✗ FAIL"
            if has_nan:
                all_ok = False
            rows.append(["Weight Grad", "(3072, 768)", status])
            
            has_nan = check_tensor_for_nan(linear.bias.grad, "bias grad", verbose=False)
            if has_nan:
                debug_info.append(f"  ✗ Bias gradient contains NaN/Inf")
                if linear.bias.grad is not None:
                    debug_info.append(f"    Bias grad stats: min={linear.bias.grad.min().item():.6f}, max={linear.bias.grad.max().item():.6f}")
            status = "✓ OK" if not has_nan else "✗ FAIL"
            if has_nan:
                all_ok = False
            rows.append(["Bias Grad", "(3072,)", status])
        except Exception as e:
            debug_info.append(f"  ✗ Backward pass failed with exception: {e}")
            rows.append(["Weight Grad", "ERROR", "✗ FAIL"])
            rows.append(["Bias Grad", "ERROR", "✗ FAIL"])
            all_ok = False
    
    print_table(
        ["Component", "Shape", "Status"],
        rows,
        "Test 3: Linear Layer Check"
    )
    
    # 如果有错误，打印详细debug信息
    if not all_ok and debug_info:
        print("\nDetailed Debug Information:")
        print("-" * 80)
        for info in debug_info:
            print(info)
    
    return all_ok

def test_training_step():
    """测试训练步骤"""
    device = torch.device('cas_npu:0')
    
    model = nn.Linear(10, 5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    x = torch.randn(2, 10, device=device)
    target = torch.randn(2, 5, device=device)
    
    rows = []
    all_ok = True
    
    # 前向传播
    output = model(x)
    has_nan = check_tensor_for_nan(output, "forward output", verbose=False)
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["Forward", "(2, 5)", status])
    
    # 损失计算
    loss = nn.functional.mse_loss(output, target)
    has_nan = check_tensor_for_nan(loss, "loss", verbose=False)
    loss_val = f"{loss.item():.6f}" if not has_nan else "NaN/Inf"
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["Loss", loss_val, status])
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    has_nan = check_tensor_for_nan(model.weight.grad, "weight grad", verbose=False)
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["Weight Grad", "(5, 10)", status])
    
    has_nan = check_tensor_for_nan(model.bias.grad, "bias grad", verbose=False)
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["Bias Grad", "(5,)", status])
    
    # 优化器更新
    optimizer.step()
    
    has_nan = check_tensor_for_nan(model.weight, "weight after update", verbose=False)
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["Weight (updated)", "(5, 10)", status])
    
    has_nan = check_tensor_for_nan(model.bias, "bias after update", verbose=False)
    status = "✓ OK" if not has_nan else "✗ FAIL"
    if has_nan:
        all_ok = False
    rows.append(["Bias (updated)", "(5,)", status])
    
    print_table(
        ["Step", "Value/Shape", "Status"],
        rows,
        "Test 4: Training Step Check"
    )
    return all_ok

def main():
    print("=" * 80)
    print("NaN Diagnosis Tool")
    print("=" * 80)
    
    results = []
    all_passed = True
    
    try:
        passed = test_memory_initialization()
        results.append(["Test 1", "Memory Initialization", "✓ PASS" if passed else "✗ FAIL"])
        if not passed:
            all_passed = False
    except Exception as e:
        results.append(["Test 1", "Memory Initialization", f"✗ ERROR: {str(e)[:30]}"])
        all_passed = False
    
    try:
        passed = test_basic_operations()
        results.append(["Test 2", "Basic Operations", "✓ PASS" if passed else "✗ FAIL"])
        if not passed:
            all_passed = False
    except Exception as e:
        results.append(["Test 2", "Basic Operations", f"✗ ERROR: {str(e)[:30]}"])
        all_passed = False
    
    try:
        passed = test_linear_layer()
        results.append(["Test 3", "Linear Layer", "✓ PASS" if passed else "✗ FAIL"])
        if not passed:
            all_passed = False
    except Exception as e:
        results.append(["Test 3", "Linear Layer", f"✗ ERROR: {str(e)[:30]}"])
        all_passed = False
    
    try:
        passed = test_training_step()
        results.append(["Test 4", "Training Step", "✓ PASS" if passed else "✗ FAIL"])
        if not passed:
            all_passed = False
    except Exception as e:
        results.append(["Test 4", "Training Step", f"✗ ERROR: {str(e)[:30]}"])
        all_passed = False
    
    print_table(
        ["ID", "Test Name", "Result"],
        results,
        "\nSummary"
    )
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

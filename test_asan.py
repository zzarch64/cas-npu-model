#!/usr/bin/env python
"""
AddressSanitizer 测试脚本
只测试 masked_fill_ 相关的操作，避免加载完整模型
"""
import torch
import sys
import os

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入cas_npu扩展
try:
    import cas_npu
    print("✓ CAS-NPU extension imported successfully")
except ImportError as e:
    print(f"✗ Failed to import CAS-NPU extension: {e}")
    sys.exit(1)

device = torch.device('privateuseone:0')
print(f"Using device: {device}")

# 测试 1: 简单的 masked_fill_
print("\n=== Test 1: Simple masked_fill_ ===")
try:
    x = torch.ones((2, 3), dtype=torch.float32, device=device)
    mask = torch.tensor([[True, False, True], [False, True, False]], device=device)
    print(f"Before: {x.cpu()}")
    x.masked_fill_(mask, -1.0)
    print(f"After: {x.cpu()}")
    print("✓ Test 1 passed")
except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

# 测试 2: 模拟 attention_mask 处理
print("\n=== Test 2: Attention mask processing ===")
try:
    seq_len = 10
    batch_size = 1
    
    # 创建 attention_mask
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device=device)
    print(f"attention_mask shape: {attention_mask.shape}")
    
    # 创建 causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    print(f"causal_mask shape: {causal_mask.shape}")
    
    # 合并 masks（这是 transformers 中的操作）
    combined_mask = torch.full((batch_size, 1, seq_len, seq_len), float('-inf'), device=device)
    print(f"combined_mask shape: {combined_mask.shape}, dtype: {combined_mask.dtype}")
    
    # 这一步会调用 masked_fill_
    print("Calling masked_fill_...")
    combined_mask = combined_mask.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
    print(f"After masked_fill_: shape={combined_mask.shape}")
    print(f"Sample values: {combined_mask[0, 0, :3, :3].cpu()}")
    print("✓ Test 2 passed")
except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

# 测试 3: 多次调用 masked_fill_
print("\n=== Test 3: Multiple masked_fill_ calls ===")
try:
    x = torch.ones((5, 5), dtype=torch.float32, device=device)
    for i in range(10):
        mask = torch.rand(5, 5, device=device) > 0.5
        x.masked_fill_(mask, float(i))
    print(f"After 10 masked_fill_ calls: {x.cpu()}")
    print("✓ Test 3 passed")
except Exception as e:
    print(f"✗ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()

# 测试 4: 不同大小的 tensor
print("\n=== Test 4: Different tensor sizes ===")
try:
    sizes = [(1, 1), (10, 10), (100, 100), (1000, 1000)]
    for size in sizes:
        x = torch.ones(size, dtype=torch.float32, device=device)
        mask = torch.rand(size, device=device) > 0.5
        x.masked_fill_(mask, -1.0)
        print(f"  Size {size}: OK")
    print("✓ Test 4 passed")
except Exception as e:
    print(f"✗ Test 4 failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== All tests completed ===")

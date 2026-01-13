# 内存问题分析与修复

## 问题总结

在运行 Qwen 模型推理时，当传入 `attention_mask` 参数后，程序在生成阶段崩溃，报错：
```
free(): invalid pointer
invalid fastbin entry (free)
```

## 根本原因

### aligned_alloc 的问题

**原始实现**：
```cpp
size_t aligned_size = ((size + 63) / 64) * 64;
data = aligned_alloc(64, aligned_size);
```

**问题**：
1. **内存大小不匹配**：实际分配 `aligned_size` 字节，但 PyTorch 期望 `size` 字节
2. **内存对齐不一致**：强制 64 字节对齐与 PyTorch 的期望不一致
3. **兼容性问题**：与 PyTorch 的 CPU allocator 行为不一致

### 为什么 attention_mask 会触发问题？

1. **频繁的内存操作**：`attention_mask` 处理涉及多次 `masked_fill_` 调用
2. **非连续内存**：`attention_mask` 可能涉及非连续的内存布局
3. **内存对齐要求**：64 字节对齐可能过于严格

## 修复方案

### 改为使用标准的 malloc

**新实现**：
```cpp
void* data = malloc(size);
memset(data, 0, size);
allocations[data] = size;
```

**优势**：
- ✅ 与 PyTorch 保持一致
- ✅ 内存大小匹配
- ✅ 自然对齐（通常 8 或 16 字节，足够使用）
- ✅ 代码更简单

### 为什么 malloc 足够？

1. PyTorch 的 CPU 后端使用标准的 `malloc`
2. 现代 CPU 通常只需要 8 或 16 字节对齐
3. 如果确实需要对齐，可以在特定操作中处理

## 测试验证

### 简单测试

```python
import torch
import echo_npu

device = torch.device('privateuseone:0')
x = torch.ones((2, 3), device=device)
mask = torch.tensor([[True, False, True], [False, True, False]], device=device)
x.masked_fill_(mask, -1.0)
print("Result:", x.cpu())
```

### attention_mask 测试

```python
seq_len = 10
batch_size = 1
attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device=device)
causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
combined_mask = torch.full((batch_size, 1, seq_len, seq_len), float('-inf'), device=device)
combined_mask = combined_mask.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
```

## 进一步优化建议

### 如果确实需要对齐

```cpp
EchoNpuError echoNpuMallocAligned(void** ptr, size_t size, size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return ECHO_NPU_ERROR_INVALID_VALUE;
    }
    
    int err = posix_memalign(ptr, alignment, size);
    if (err != 0) {
        return ECHO_NPU_ERROR_OUT_OF_MEMORY;
    }
    
    memset(*ptr, 0, size);
    return ECHO_NPU_SUCCESS;
}
```

---

**修改日期**：2026-01-10  
**修改文件**：`runtime/cmodel/simulator.cpp`  
**状态**：✅ 已修复

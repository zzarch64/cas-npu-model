# 内存分配修复分析：从 aligned_alloc 改为 malloc

## 问题分析

### 为什么添加 attention_mask 后出现内存异常？

#### 1. aligned_alloc 的问题

**原始实现**：
```cpp
size_t aligned_size = ((size + 63) / 64) * 64;
data = aligned_alloc(64, aligned_size);
actual_size = aligned_size;  // 实际分配的大小可能大于请求的 size
```

**潜在问题**：

1. **内存大小不匹配**：
   - 请求分配 `size` 字节
   - 实际分配 `aligned_size` 字节（向上取整到64的倍数）
   - PyTorch 期望的内存大小是 `size`，但我们分配了 `aligned_size`
   - 在 `echoNpuMemcpy` 中，我们使用 `size` 进行拷贝，这是正确的
   - 但在某些情况下，如果 PyTorch 内部期望访问 `size` 大小的内存，可能会有问题

2. **内存对齐不一致**：
   - PyTorch 的 CPU allocator 使用 `malloc`，自然对齐到平台要求（通常是8或16字节）
   - 我们强制使用64字节对齐，这可能与 PyTorch 的期望不一致
   - 当 tensor 在 CPU 和 NPU 之间移动时，对齐方式不同可能导致问题

3. **内存初始化范围**：
   - 我们初始化了整个 `aligned_size` 字节
   - 但 PyTorch 可能只期望 `size` 字节被初始化
   - 虽然这通常不会直接导致问题，但可能在某些边界情况下造成混淆

#### 2. attention_mask 的特殊性

**为什么 attention_mask 会触发问题？**

1. **频繁的内存操作**：
   - `attention_mask` 的处理涉及多次 `masked_fill_` 调用
   - 每次调用都需要在 CPU 和 NPU 之间拷贝数据
   - 如果内存分配/释放不匹配，多次操作会累积错误

2. **非连续内存**：
   - `attention_mask` 可能涉及非连续的内存布局
   - `aligned_alloc` 分配的内存布局可能与 PyTorch 期望的不一致
   - 在 `_copy_from` 中处理非连续内存时，可能出现问题

3. **内存对齐要求**：
   - `masked_fill_` 操作可能对内存对齐有特定要求
   - 64字节对齐可能过于严格，导致某些操作失败

## 修复方案

### 改为使用标准的 malloc

**新实现**：
```cpp
void* data = malloc(size);
memset(data, 0, size);
allocations[data] = size;  // 记录实际分配的大小（现在是 size）
```

**优势**：

1. **与 PyTorch 一致**：
   - PyTorch 的 CPU allocator 使用 `malloc`
   - 使用相同的分配方式可以避免兼容性问题

2. **内存大小匹配**：
   - 分配的大小就是请求的 `size`
   - 不会有多余的内存，避免大小不匹配问题

3. **自然对齐**：
   - `malloc` 自然对齐到平台要求（通常是8或16字节）
   - 这通常足够满足大多数操作的需求
   - 如果确实需要64字节对齐，可以在特定操作中处理

4. **简化代码**：
   - 不需要计算 `aligned_size`
   - 不需要区分 Windows 和 Linux 的实现
   - 代码更简单，更容易维护

### 为什么 malloc 足够？

1. **PyTorch 使用 malloc**：
   - PyTorch 的 CPU 后端使用标准的 `malloc`
   - 大多数操作不需要特殊的对齐要求

2. **自然对齐足够**：
   - 现代 CPU 通常只需要8或16字节对齐
   - 64字节对齐通常是为了 SIMD 优化，但不是必需的

3. **如果需要对齐**：
   - 可以在特定操作中手动对齐
   - 或者使用 `posix_memalign` 在需要时分配对齐的内存

## 测试建议

### 1. 简单测试

```python
import torch
import sys
sys.path.insert(0, '.')
import echo_npu

device = torch.device('privateuseone:0')
x = torch.ones((2, 3), device=device)
mask = torch.tensor([[True, False, True], [False, True, False]], device=device)
x.masked_fill_(mask, -1.0)
print("Result:", x.cpu())
```

### 2. attention_mask 测试

```python
seq_len = 10
batch_size = 1
attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device=device)
causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
combined_mask = torch.full((batch_size, 1, seq_len, seq_len), float('-inf'), device=device)
combined_mask = combined_mask.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
print("Success!")
```

### 3. 完整模型测试

```bash
python examples/qwen_inference.py --prompt "讲个笑话"
```

## 如果问题仍然存在

如果改为 `malloc` 后问题仍然存在，可能的原因：

### 1. 其他内存问题

- `echoNpuMemcpy` 中的大小计算错误
- `_copy_from` 中的内存拷贝逻辑问题
- 内存释放时机不对

### 2. 线程安全问题

- `allocations` map 的线程安全性
- 多线程环境下的内存操作

### 3. PyTorch 版本问题

- PyTorch 2.8.0 的新行为
- functorch 的兼容性问题

## 进一步优化建议

### 如果确实需要对齐

如果某些操作确实需要64字节对齐，可以：

1. **按需对齐**：
```cpp
EchoNpuError echoNpuMallocAligned(void** ptr, size_t size, size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return ECHO_NPU_ERROR_INVALID_VALUE;  // alignment 必须是2的幂
    }
    
    // 使用 posix_memalign 分配对齐的内存
    int err = posix_memalign(ptr, alignment, size);
    if (err != 0) {
        return ECHO_NPU_ERROR_OUT_OF_MEMORY;
    }
    
    memset(*ptr, 0, size);
    return ECHO_NPU_SUCCESS;
}
```

2. **在特定操作中处理**：
   - 只在需要对齐的操作中使用对齐分配
   - 大多数操作使用标准的 `malloc`

### 内存池优化

如果性能是问题，可以考虑实现内存池：

```cpp
class MemoryPool {
    std::vector<void*> free_blocks;
    std::mutex mutex;
    
public:
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        // 从池中获取或分配新内存
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex);
        // 返回到池中
    }
};
```

## 总结

1. ✅ **已修复**：将 `aligned_alloc` 改为标准的 `malloc`
2. ✅ **原因**：`aligned_alloc` 导致内存大小不匹配和对齐不一致
3. ✅ **优势**：与 PyTorch 保持一致，简化代码，避免兼容性问题
4. ⏳ **待测试**：验证修复是否解决了内存异常问题

---

**修改日期**：2026-01-10  
**修改文件**：`runtime/cmodel/simulator.cpp`  
**修改内容**：`echoNpuMalloc` 和 `echoNpuFree` 函数

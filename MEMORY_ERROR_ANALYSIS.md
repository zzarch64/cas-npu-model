# 内存错误分析报告：masked_fill_ 崩溃问题

## 问题总结

在运行 Qwen 模型推理时，当传入 `attention_mask` 参数后，程序在生成阶段崩溃，报错：
```
free(): invalid pointer
invalid fastbin entry (free)
```

## 调查过程

### 1. 问题复现

- **无 attention_mask**：程序运行超时（120秒+），但**不崩溃**
- **有 attention_mask**：程序在几秒内崩溃，报告内存错误

### 2. 关键发现

#### 测试 1：简单的 masked_fill_ 操作

```python
device = torch.device('privateuseone:0')
x = torch.ones((2, 3), device=device)
mask = torch.tensor([[True, False, True], [False, True, False]], device=device)
x.masked_fill_(mask, -1.0)
print("Result:", x.cpu())  # ✅ 成功，无内存错误
```

**结论**：简单情况下 `masked_fill_` 工作正常。

#### 测试 2：完整模型生成

```python
# 传入 attention_mask
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,  # ← 问题触发点
    generation_config=generation_config,
)
```

**结论**：在复杂的模型生成过程中崩溃。

### 3. GDB 堆栈分析

崩溃发生在：
```
#10 __posix_memalign (size=1, alignment=64, memptr=0x7fffffff9eb8)
#29 at::_ops::all::call(at::Tensor const&)
```

**关键发现**：
- 崩溃并非直接发生在 `masked_fill_`，而是在后续的 `all` 操作中
- 这说明 `masked_fill_` 或相关操作破坏了内存元数据
- 当后续操作尝试分配内存时，检测到内存已被破坏

### 4. Valgrind 分析

由于 Valgrind 在 Python/PyTorch 程序上产生大量噪音，难以直接定位问题。

## 可能的原因分析

### 原因 1：cpu_fallback 与 functorch 冲突

**假设**：PyTorch 的 functorch dynamic layer 在处理 inplace 操作时，与我们的 `cpu_fallback` 机制冲突，导致内存管理错误。

**证据**：
- GDB 堆栈显示 `at::functorch::` 相关调用
- 简单的 `masked_fill_` 正常，但在复杂生成中崩溃
- PyTorch 2.8.0 可能引入了新的 functorch 行为

**反驳**：
- 网上没有其他人报告类似问题
- 简单测试通过说明基础功能正常

### 原因 2：_copy_from 实现问题

**假设**：`_copy_from` 在处理某些特殊情况（如非 contiguous tensor）时出错。

**证据**：
- `masked_fill_` 需要多次调用 `_copy_from`
- 复杂的内存拷贝可能出现边界问题

**待验证**：
- 检查 `_copy_from` 中的 size 计算
- 检查 contiguous 处理逻辑

### 原因 3：内存分配器问题

**假设**：`casNpuMalloc` / `casNpuFree` 在某些情况下破坏了内存元数据。

**证据**：
- 错误信息 "invalid fastbin entry" 指向 malloc/free
- 问题出现在 `__posix_memalign`

**待验证**：
- 检查 `aligned_alloc` 的 alignment 参数
- 检查 `allocations` map 的线程安全性

### 原因 4：attention_mask 的特殊处理

**假设**：transformers 对 attention_mask 的处理方式触发了特殊的代码路径。

**证据**：
- 不传 attention_mask 不崩溃
- 传 attention_mask 崩溃

**可能路径**：
```
attention_mask -> causal mask 合并 -> masked_fill_ -> 内存错误
```

## 尝试的解决方案

### 方案 1：自定义 masked_fill_ 实现 ❌

**尝试**：
```cpp
at::Tensor& cas_npu_masked_fill_Scalar(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {
    at::Tensor result = at::masked_fill(self, mask, value);
    self.copy_(result);
    return self;
}
```

**结果**：仍然崩溃。

**分析**：自定义实现反而引入了更多的 `copy_` 调用，可能加剧问题。

### 方案 2：确保 contiguous ❌

**尝试**：在操作前将所有 tensor 变成 contiguous。

**结果**：仍然崩溃。

### 方案 3：使用纯 cpu_fallback ❌

**尝试**：所有 masked_fill 相关操作都使用 `cpu_fallback`。

**结果**：仍然崩溃。

### 方案 4：Python 层拦截 ❌

**尝试**：
```python
_original_masked_fill_ = torch.Tensor.masked_fill_
def _patched_masked_fill_(self, mask, value):
    if self.device.type == 'privateuseone':
        result = torch.masked_fill(self, mask, value)
        self.copy_(result)
        return self
    return _original_masked_fill_(self, mask, value)
torch.Tensor.masked_fill_ = _patched_masked_fill_
```

**结果**：没有效果（因为 C++ 层直接调用 aten::masked_fill_）。

## 进一步调试建议

### 方法 1：使用 AddressSanitizer

AddressSanitizer 比 Valgrind 更快，且报告更清晰。

```bash
# 重新编译扩展
export CFLAGS="-fsanitize=address -g -O0"
export CXXFLAGS="-fsanitize=address -g -O0"
export LDFLAGS="-fsanitize=address"

python setup.py clean --all
python setup.py build_ext --inplace

# 运行测试
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=1
python examples/qwen_inference.py --prompt "讲个笑话"
```

### 方法 2：添加内存调试代码

在 `simulator.cpp` 中添加：

```cpp
// 在 casNpuMalloc 中
CasNpuError casNpuMalloc(void** ptr, size_t size) {
    // 添加调试信息
    fprintf(stderr, "[DEBUG] casNpuMalloc: size=%zu\n", size);
    
    // ... 原有代码 ...
    
    fprintf(stderr, "[DEBUG] casNpuMalloc: ptr=%p, actual_size=%zu\n", data, actual_size);
    return CAS_NPU_SUCCESS;
}

// 在 casNpuFree 中
CasNpuError casNpuFree(void* ptr) {
    fprintf(stderr, "[DEBUG] casNpuFree: ptr=%p\n", ptr);
    
    // 检查是否在 allocations 中
    {
        std::lock_guard<std::mutex> lock(allocation_mutex);
        auto it = allocations.find(ptr);
        if (it == allocations.end()) {
            fprintf(stderr, "[ERROR] Attempt to free untracked pointer: %p\n", ptr);
            // 打印当前所有已分配的指针
            fprintf(stderr, "[DEBUG] Currently allocated pointers:\n");
            for (const auto& pair : allocations) {
                fprintf(stderr, "  %p -> %zu bytes\n", pair.first, pair.second);
            }
            return CAS_NPU_ERROR_INVALID_VALUE;
        }
        fprintf(stderr, "[DEBUG] Freeing %zu bytes at %p\n", it->second, ptr);
    }
    
    // ... 原有代码 ...
}
```

### 方法 3：检查 alignment 问题

在 `simulator.cpp` 的 `casNpuMalloc` 中：

```cpp
// 当前实现
size_t aligned_size = ((size + 63) / 64) * 64;
data = aligned_alloc(64, aligned_size);

// 添加检查
if ((uintptr_t)data % 64 != 0) {
    fprintf(stderr, "[ERROR] Allocated memory not aligned! ptr=%p\n", data);
}
```

### 方法 4：最小化复现

创建一个只包含 attention mask 处理的最小示例：

```python
import torch
import sys
sys.path.insert(0, '.')
import cas_npu

device = torch.device('privateuseone:0')

# 模拟 attention_mask 的处理
seq_len = 10
batch_size = 1

# 创建 attention_mask
attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device=device)

# 创建 causal mask
causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

# 合并 masks（这是 transformers 中的操作）
combined_mask = torch.full((batch_size, 1, seq_len, seq_len), float('-inf'), device=device)
combined_mask = combined_mask.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)

print("Combined mask shape:", combined_mask.shape)
print("Success!")
```

### 方法 5：比较 PyTorch 版本

```bash
# 测试不同 PyTorch 版本
conda create -n test_pt20 python=3.9
conda activate test_pt20
pip install torch==2.0.0

# 运行测试
python examples/qwen_inference.py --prompt "讲个笑话"
```

## 临时解决方案

在问题未解决前，可以：

### 选项 1：不使用 attention_mask

修改 `qwen_inference.py`：
```python
# 不传递 attention_mask
outputs = model.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    # attention_mask=None,  # 不传递
)
```

**警告**：模型可能会有警告，且在某些情况下结果可能不准确。

### 选项 2：使用 CPU 后端

```python
# 使用 CPU 进行推理
device = torch.device('cpu')
```

### 选项 3：降级 PyTorch

```bash
pip install torch==2.1.0
```

## 后续行动计划

1. ✅ **完成**：创建 Valgrind 调试文档
2. ⏳ **进行中**：使用 AddressSanitizer 重新测试
3. ⏳ **待办**：添加详细的内存调试日志
4. ⏳ **待办**：创建最小化复现示例
5. ⏳ **待办**：测试不同 PyTorch 版本
6. ⏳ **待办**：向 PyTorch 社区报告（如果确认是 PyTorch 问题）

## 参考资料

- [PyTorch Custom Backend Guide](https://pytorch.org/tutorials/advanced/extend_dispatcher.html)
- [Functorch Documentation](https://pytorch.org/functorch/stable/)
- [Memory Debugging with Valgrind](./VALGRIND_DEBUG_GUIDE.md)
- [GDB Debugging Guide](./GDB_DEBUG_GUIDE.md)

---

**创建日期**：2026-01-10  
**最后更新**：2026-01-10  
**状态**：调查中

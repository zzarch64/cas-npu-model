# Valgrind 内存调试指南

本文档介绍如何使用 Valgrind 调试 C/C++ 扩展的内存问题，特别是 PyTorch 自定义后端扩展。

## 目录

1. [Valgrind 简介](#valgrind-简介)
2. [安装 Valgrind](#安装-valgrind)
3. [基本使用](#基本使用)
4. [常见内存错误类型](#常见内存错误类型)
5. [调试 PyTorch 扩展](#调试-pytorch-扩展)
6. [解读 Valgrind 输出](#解读-valgrind-输出)
7. [常见问题](#常见问题)
8. [最佳实践](#最佳实践)

---

## Valgrind 简介

Valgrind 是一个强大的内存调试和性能分析工具，可以检测：
- 内存泄漏（memory leaks）
- 无效内存访问（invalid memory access）
- 使用未初始化的内存（use of uninitialized memory）
- 重复释放内存（double free）
- 内存越界访问（buffer overflow）

### 工作原理

Valgrind 通过在虚拟环境中运行程序，监控所有内存操作。这会使程序运行速度降低 10-50 倍。

---

## 安装 Valgrind

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install valgrind
```

### CentOS/RHEL
```bash
sudo yum install valgrind
```

### 验证安装
```bash
valgrind --version
```

---

## 基本使用

### 1. 最简单的用法

```bash
valgrind ./your_program
```

### 2. 检测内存泄漏

```bash
valgrind --leak-check=full ./your_program
```

### 3. 显示所有类型的泄漏

```bash
valgrind --leak-check=full --show-leak-kinds=all ./your_program
```

### 4. 追踪未初始化内存的来源

```bash
valgrind --track-origins=yes ./your_program
```

### 5. 完整的调试命令

```bash
valgrind \
  --leak-check=full \
  --show-leak-kinds=all \
  --track-origins=yes \
  --verbose \
  --log-file=valgrind_output.log \
  ./your_program
```

### 6. Python 程序调试

```bash
valgrind --leak-check=full python your_script.py
```

---

## 常见内存错误类型

### 1. Invalid read/write（无效读写）

```
==12345== Invalid read of size 4
==12345==    at 0x12345678: function_name (file.cpp:42)
```

**原因**：
- 访问已释放的内存
- 数组越界
- 空指针解引用

**示例代码**：
```cpp
int* ptr = new int[10];
delete[] ptr;
int value = ptr[0];  // Invalid read!
```

### 2. Memory leak（内存泄漏）

```
==12345== 100 bytes in 1 blocks are definitely lost
==12345==    at 0x4C2DB8F: malloc (vg_replace_malloc.c:299)
==12345==    by 0x12345678: function_name (file.cpp:42)
```

**原因**：
- 分配的内存没有释放
- 丢失了指向分配内存的指针

**示例代码**：
```cpp
void leak() {
    int* ptr = new int[100];
    // 忘记 delete[] ptr;
}
```

### 3. Invalid free（无效释放）

```
==12345== Invalid free() / delete / delete[] / realloc()
==12345==    at 0x4C2EDEB: free (vg_replace_malloc.c:530)
==12345==    by 0x12345678: function_name (file.cpp:42)
```

**原因**：
- 重复释放同一块内存（double free）
- 释放未分配的内存
- 释放栈上的内存

**示例代码**：
```cpp
int* ptr = new int;
delete ptr;
delete ptr;  // Double free!
```

### 4. Use of uninitialized value（使用未初始化的值）

```
==12345== Conditional jump or move depends on uninitialised value(s)
==12345==    at 0x12345678: function_name (file.cpp:42)
```

**原因**：
- 使用未初始化的变量
- 读取未初始化的内存

**示例代码**：
```cpp
int x;
if (x > 10) {  // x 未初始化
    // ...
}
```

---

## 调试 PyTorch 扩展

### 挑战

PyTorch 扩展的调试比普通 C++ 程序更复杂：

1. **Python 解释器噪音**：Python 解释器本身会产生很多 Valgrind 报告
2. **PyTorch 库噪音**：PyTorch/libtorch 也会产生大量报告
3. **运行缓慢**：Valgrind 会使程序运行非常慢

### 策略

#### 1. 使用 Python 的 Valgrind suppressions

创建 `python.supp` 文件来抑制 Python 相关的误报：

```bash
# 获取 Python suppressions
wget https://github.com/python/cpython/raw/main/Misc/valgrind-python.supp

# 使用 suppressions
valgrind --suppressions=valgrind-python.supp python your_script.py
```

#### 2. 只关注特定错误

```bash
# 只检测无效内存访问，不检测泄漏
valgrind --leak-check=no python your_script.py

# 限制错误数量
valgrind --error-limit=yes python your_script.py
```

#### 3. 使用日志文件

```bash
valgrind --log-file=valgrind_%p.log python your_script.py
```

其中 `%p` 会被替换为进程 ID。

#### 4. 针对性测试

创建最小可复现示例（minimal reproducible example）：

```python
# test_minimal.py
import torch
import sys
sys.path.insert(0, '.')
import cas_npu

device = torch.device('privateuseone:0')
x = torch.ones((2, 3), device=device)
mask = torch.tensor([[True, False, True], [False, True, False]], device=device)
x.masked_fill_(mask, -1.0)
print("Result:", x.cpu())
```

```bash
valgrind --leak-check=full python test_minimal.py 2>&1 | grep -i "cas_npu\|invalid\|LEAK"
```

---

## 解读 Valgrind 输出

### 输出格式

```
==12345== Invalid read of size 4
==12345==    at 0x4C2DB8F: function_name (file.cpp:42)
==12345==    by 0x4C2DC9A: caller_function (file.cpp:100)
==12345==    by 0x4C2DE1B: main (main.cpp:10)
==12345==  Address 0x5204040 is 0 bytes after a block of size 16 free'd
==12345==    at 0x4C2EDEB: operator delete(void*) (vg_replace_malloc.c:595)
==12345==    by 0x4C2DC9A: function_name (file.cpp:50)
```

### 关键信息

1. **`==12345==`**：进程 ID
2. **错误类型**：`Invalid read of size 4`
3. **发生位置**：`at 0x4C2DB8F: function_name (file.cpp:42)`
4. **调用栈**：`by` 开头的行显示调用链
5. **内存地址信息**：`Address 0x5204040`
6. **分配/释放位置**：显示内存的分配或释放位置

### 泄漏总结

```
==12345== LEAK SUMMARY:
==12345==    definitely lost: 100 bytes in 1 blocks
==12345==    indirectly lost: 200 bytes in 5 blocks
==12345==      possibly lost: 50 bytes in 2 blocks
==12345==    still reachable: 1,000 bytes in 10 blocks
==12345==         suppressed: 0 bytes in 0 blocks
```

- **definitely lost**：确定的内存泄漏，必须修复
- **indirectly lost**：间接泄漏，通常随着 definitely lost 一起修复
- **possibly lost**：可能的泄漏，需要检查
- **still reachable**：程序结束时仍可达的内存，通常不是问题
- **suppressed**：被抑制的错误

---

## 常见问题

### Q1: Valgrind 运行太慢怎么办？

**A**: 
- 使用最小测试用例
- 只运行关键部分代码
- 使用 `--leak-check=no` 跳过泄漏检测
- 增加超时时间

### Q2: 输出太多无关信息怎么办？

**A**:
- 使用 suppressions 文件
- 使用 `grep` 过滤输出
- 使用 `--error-limit=yes`
- 专注于特定函数或文件

### Q3: Python 程序产生大量误报怎么办？

**A**:
- 使用 Python 官方的 suppressions 文件
- 关注自己代码的错误（通过文件名过滤）
- 使用 `--show-leak-kinds=definite` 只显示确定的泄漏

### Q4: 如何定位自己扩展的问题？

**A**:
```bash
valgrind python script.py 2>&1 | grep "your_extension_name\|your_cpp_file"
```

### Q5: Valgrind 报告 "invalid fastbin entry (free)" 怎么办？

**A**: 这通常表示：
- 内存被破坏（heap corruption）
- 在之前的某个操作中发生了越界写入
- 需要回溯找到真正的错误源

使用 gdb 配合 Valgrind：
```bash
valgrind --vgdb=yes --vgdb-error=0 python script.py
# 在另一个终端
gdb -ex "target remote | vgdb"
```

---

## 最佳实践

### 1. 开发阶段

- **经常运行 Valgrind**：不要等到出现问题才运行
- **修复所有错误**：包括 "possibly lost"
- **编写单元测试**：为每个功能编写 Valgrind 测试

### 2. 代码编写

```cpp
// ✅ 好的做法
void good_practice() {
    int* ptr = new int[100];
    // 使用 ptr
    delete[] ptr;
    ptr = nullptr;  // 防止悬空指针
}

// ❌ 坏的做法
void bad_practice() {
    int* ptr = new int[100];
    // 使用 ptr
    delete[] ptr;
    // 忘记设置为 nullptr
    // 后续可能误用 ptr
}
```

### 3. 智能指针

使用智能指针可以避免很多内存问题：

```cpp
#include <memory>

// 使用 unique_ptr
std::unique_ptr<int[]> ptr = std::make_unique<int[]>(100);
// 自动释放，无需手动 delete

// 使用 shared_ptr
std::shared_ptr<Data> data = std::make_shared<Data>();
// 引用计数自动管理
```

### 4. RAII (Resource Acquisition Is Initialization)

```cpp
class Buffer {
private:
    float* data;
    size_t size;
    
public:
    Buffer(size_t n) : size(n) {
        data = new float[n];
    }
    
    ~Buffer() {
        delete[] data;
    }
    
    // 禁止拷贝
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
```

### 5. 调试技巧

```cpp
// 添加调试宏
#ifdef DEBUG_MEMORY
#define DEBUG_ALLOC(ptr, size) \
    fprintf(stderr, "ALLOC: %p, size=%zu at %s:%d\n", ptr, size, __FILE__, __LINE__)
#define DEBUG_FREE(ptr) \
    fprintf(stderr, "FREE: %p at %s:%d\n", ptr, __FILE__, __LINE__)
#else
#define DEBUG_ALLOC(ptr, size)
#define DEBUG_FREE(ptr)
#endif

void* my_malloc(size_t size) {
    void* ptr = malloc(size);
    DEBUG_ALLOC(ptr, size);
    return ptr;
}

void my_free(void* ptr) {
    DEBUG_FREE(ptr);
    free(ptr);
}
```

---

## 实战示例：调试 CAS-NPU 扩展

### 问题描述

在运行 Qwen 模型推理时遇到 `free(): invalid pointer` 错误。

### 调试步骤

#### 1. 创建最小测试用例

```python
# test_masked_fill.py
import torch
import sys
sys.path.insert(0, '.')
import cas_npu

device = torch.device('privateuseone:0')
print('Testing masked_fill_...')

x = torch.ones((2, 3), dtype=torch.float32, device=device)
mask = torch.tensor([[True, False, True], [False, True, False]], device=device)

try:
    x.masked_fill_(mask, -1.0)
    print('Result:', x.cpu())
    print('SUCCESS!')
except Exception as e:
    print(f'ERROR: {e}')
```

#### 2. 运行 Valgrind

```bash
valgrind --leak-check=full \
         --track-origins=yes \
         --log-file=valgrind_masked_fill.log \
         python test_masked_fill.py
```

#### 3. 分析输出

```bash
# 查找与我们代码相关的错误
grep -i "cas_npu\|masked_fill" valgrind_masked_fill.log

# 查找内存错误
grep -i "invalid\|leak" valgrind_masked_fill.log | head -50
```

#### 4. 定位问题

假设发现：
```
==12345== Invalid write of size 4
==12345==    at 0x4E8F123: casNpuMemcpy (simulator.cpp:140)
==12345==    by 0x4E8F456: cas_npu_copy_from (cas_npu_ops.cpp:459)
```

这表明问题在 `casNpuMemcpy` 中。

#### 5. 修复代码

检查 `simulator.cpp:140` 附近的代码：

```cpp
// 可能的问题：size 计算错误
CasNpuError casNpuMemcpy(void* dst, const void* src, size_t size, CasNpuMemcpyKind kind) {
    if (size > 0) {
        // 确保不会越界
        memcpy(dst, src, size);  // ← 检查 dst 和 src 的实际大小
    }
    return CAS_NPU_SUCCESS;
}
```

#### 6. 验证修复

再次运行 Valgrind，确认错误已消失。

---

## 替代工具

当 Valgrind 不够用时，可以尝试：

### 1. AddressSanitizer (ASan)

编译时启用：
```bash
CFLAGS="-fsanitize=address -g" python setup.py build_ext --inplace
export ASAN_OPTIONS=detect_leaks=1
python your_script.py
```

**优点**：
- 比 Valgrind 快得多
- 检测更多类型的错误

**缺点**：
- 需要重新编译
- 与某些库不兼容

### 2. GDB

用于更精确的调试：
```bash
gdb --args python your_script.py
(gdb) run
(gdb) bt  # 崩溃时查看堆栈
```

### 3. heaptrack

专门的堆内存分析工具：
```bash
heaptrack python your_script.py
heaptrack_gui heaptrack.python.12345.gz
```

---

## 总结

### Valgrind 的优势
- ✅ 无需重新编译
- ✅ 检测多种内存问题
- ✅ 详细的错误报告

### Valgrind 的局限
- ❌ 运行速度慢
- ❌ Python 程序产生大量噪音
- ❌ 不能检测所有类型的错误

### 推荐工作流

1. **开发阶段**：使用 AddressSanitizer（快速反馈）
2. **测试阶段**：使用 Valgrind（全面检查）
3. **调试阶段**：使用 GDB（精确定位）
4. **性能分析**：使用 heaptrack 或 Valgrind --tool=massif

---

## 参考资源

- [Valgrind 官方文档](https://valgrind.org/docs/manual/manual.html)
- [Valgrind Quick Start Guide](https://valgrind.org/docs/manual/quick-start.html)
- [Python Valgrind Suppressions](https://github.com/python/cpython/tree/main/Misc)
- [Google Sanitizers](https://github.com/google/sanitizers)

---

## 附录：常用命令速查

```bash
# 基础检查
valgrind ./program

# 完整内存泄漏检查
valgrind --leak-check=full --show-leak-kinds=all ./program

# 追踪未初始化内存
valgrind --track-origins=yes ./program

# Python 程序
valgrind --leak-check=full python script.py

# 输出到文件
valgrind --log-file=output.log ./program

# 使用 suppressions
valgrind --suppressions=python.supp python script.py

# GDB 集成
valgrind --vgdb=yes --vgdb-error=0 ./program
# 然后在另一个终端：gdb -ex "target remote | vgdb"

# 只检查特定错误
valgrind --leak-check=no ./program

# 限制错误报告数量
valgrind --error-limit=yes ./program
```

---

**最后更新**：2026-01-10  
**作者**：AI Assistant  
**版本**：1.0

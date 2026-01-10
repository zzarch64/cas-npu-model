# AddressSanitizer 分析报告

## 执行情况

### 编译状态
✅ **成功编译**：使用 AddressSanitizer 标志重新编译了所有 C++ 源文件

编译命令：
```bash
export CFLAGS="-fsanitize=address -g -O0"
export CXXFLAGS="-fsanitize=address -g -O0"
export LDFLAGS="-fsanitize=address"
python setup.py build_ext --inplace
```

### 运行遇到的问题

#### 问题 1: ASan 运行时库加载顺序
```
==2517240==ASan runtime does not come first in initial library list; 
you should either link runtime to your application or manually preload it with LD_PRELOAD.
```

**解决方案**：
```bash
export LD_PRELOAD=$(gcc -print-file-name=libasan.so)
```

#### 问题 2: 与 PyTorch CUDA 库冲突
```
==2517337==AddressSanitizer CHECK failed: 
../../../../src/libsanitizer/asan/asan_interceptors.cpp:335 
"((__interception::real___cxa_throw)) != (0)" (0x0, 0x0)
```

**原因**：PyTorch 的 CUDA 库（libc10_cuda.so）在初始化时抛出异常，但 AddressSanitizer 的异常拦截器未正确初始化。

**堆栈跟踪**：
```
#3 0x7fe158e76aab in c10::detail::torchCheckFail
#4 0x7fe158fadf2d in c10::cuda::(anonymous namespace)::device_count_impl
#5 0x7fe158fae076 in c10::cuda::device_count()
```

这说明问题发生在 PyTorch 检查 CUDA 设备数量时。

## 解决方案

### 方案 1: 禁用 CUDA 相关检查（推荐用于测试）

在运行前设置环境变量：
```bash
export CUDA_VISIBLE_DEVICES=""
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:handle_abort=1
export LD_PRELOAD=$(gcc -print-file-name=libasan.so)
python test_asan.py
```

### 方案 2: 使用简化的测试程序

创建了 `test_asan.py`，只测试 `masked_fill_` 相关操作，不加载完整模型：

```python
# 测试内容：
# 1. 简单的 masked_fill_ 操作
# 2. 模拟 attention_mask 处理
# 3. 多次调用 masked_fill_
# 4. 不同大小的 tensor
```

### 方案 3: 使用 AddressSanitizer 的 suppressions

创建 `asan.supp` 文件来抑制 PyTorch 相关的误报：

```
# asan.supp
interceptor_via_lib:libc10_cuda.so
interceptor_via_lib:libtorch_cpu.so
```

然后运行：
```bash
export ASAN_OPTIONS=suppressions=asan.supp:detect_leaks=1
python test_asan.py
```

## 替代调试方法

由于 AddressSanitizer 与 PyTorch CUDA 库存在兼容性问题，建议使用以下方法：

### 方法 1: 使用 GDB + AddressSanitizer

```bash
# 编译时启用 AddressSanitizer
export CFLAGS="-fsanitize=address -g -O0"
export CXXFLAGS="-fsanitize=address -g -O0"
export LDFLAGS="-fsanitize=address"
python setup.py build_ext --inplace

# 使用 GDB 运行
gdb --args python test_asan.py
(gdb) run
(gdb) bt  # 崩溃时查看堆栈
```

### 方法 2: 使用 AddressSanitizer 的符号化工具

如果程序崩溃，AddressSanitizer 会输出错误报告。使用 `asan_symbolize.py` 来符号化：

```bash
python test_asan.py 2>&1 | python asan_symbolize.py
```

### 方法 3: 使用 LeakSanitizer（LSan）

只检测内存泄漏，不检测其他错误：

```bash
export LSAN_OPTIONS=detect_leaks=1
python test_asan.py
```

## 预期输出格式

如果 AddressSanitizer 检测到问题，输出格式如下：

```
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x7f8b8c000000
READ of size 4 at 0x7f8b8c000000 thread T0
    #0 0x4e8f123 in casNpuMemcpy simulator.cpp:140
    #1 0x4e8f456 in cas_npu_copy_from cas_npu_ops.cpp:459
    #2 0x4e8f789 in cas_npu_masked_fill_Scalar cas_npu_ops.cpp:1022

0x7f8b8c000000 is located 0 bytes inside of 16-byte region [0x7f8b8c000000,0x7f8b8c000010)
freed by thread T0 here:
    #0 0x7f8b8d123456 in free (/usr/lib/x86_64-linux-gnu/libasan.so.5+0x123456)
    #1 0x4e8f789 in casNpuFree simulator.cpp:114
    #2 0x4e8fabc in ~CasNpuAllocator cas_npu_allocator.cpp:43

SUMMARY: AddressSanitizer: heap-use-after-free simulator.cpp:140 in casNpuMemcpy
```

## 关键检查点

基于之前的分析，AddressSanitizer 应该重点检查：

1. **casNpuMemcpy**：内存拷贝时是否越界
2. **casNpuMalloc/Free**：内存分配/释放是否正确配对
3. **cas_npu_copy_from**：拷贝操作是否正确处理大小
4. **masked_fill_ 实现**：是否有内存访问错误

## 下一步行动

1. ✅ **完成**：使用 AddressSanitizer 编译扩展
2. ⏳ **待办**：解决 PyTorch CUDA 库冲突问题
3. ⏳ **待办**：运行简化的测试程序
4. ⏳ **待办**：分析 AddressSanitizer 输出
5. ⏳ **待办**：根据报告修复问题

## 参考命令

### 完整调试命令（推荐）

```bash
# 1. 清理并重新编译
cd /home/zizhao.liu/code/npu_cas_extension
python setup.py clean --all
export CFLAGS="-fsanitize=address -g -O0"
export CXXFLAGS="-fsanitize=address -g -O0"
export LDFLAGS="-fsanitize=address"
python setup.py build_ext --inplace

# 2. 运行测试（禁用 CUDA）
export CUDA_VISIBLE_DEVICES=""
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:print_stats=1
export LD_PRELOAD=$(gcc -print-file-name=libasan.so)
python test_asan.py 2>&1 | tee asan_output.log

# 3. 分析输出
grep -i "error\|leak\|invalid" asan_output.log
```

### 如果遇到 CUDA 冲突

```bash
# 方法 1: 完全禁用 CUDA
export CUDA_VISIBLE_DEVICES=""
export TORCH_USE_CUDA_DSA=0

# 方法 2: 使用 CPU 后端
# 在 Python 代码中：device = torch.device('cpu')

# 方法 3: 只测试我们的扩展，不加载 PyTorch 的 CUDA 部分
# 使用 test_asan.py 而不是完整的 qwen_inference.py
```

## 注意事项

1. **性能影响**：AddressSanitizer 会使程序运行速度降低 2-3 倍
2. **内存占用**：会增加约 2-3 倍的内存使用
3. **兼容性**：某些库（如 PyTorch CUDA）可能与 AddressSanitizer 不兼容
4. **调试符号**：确保使用 `-g` 标志编译，以便获得有意义的堆栈跟踪

---

**创建日期**：2026-01-10  
**最后更新**：2026-01-10  
**状态**：遇到 PyTorch CUDA 兼容性问题，需要进一步调试

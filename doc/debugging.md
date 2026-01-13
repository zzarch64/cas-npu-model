# 调试指南

本指南介绍如何使用各种调试工具来调试 ECHO-NPU 扩展中的问题。

## 目录

1. [GDB 调试](#gdb-调试)
2. [Valgrind 内存调试](#valgrind-内存调试)
3. [AddressSanitizer](#addresssanitizer)
4. [快速参考](#快速参考)

---

## GDB 调试

### 快速开始

```bash
# 使用便捷脚本
./run_gdb.sh test/test_lenet.py

# 或直接使用 GDB
gdb --args python test/test_lenet.py
```

### 常用命令

```gdb
# 运行程序
(gdb) run

# 设置断点
(gdb) break echo_npu_add_Tensor
(gdb) break echo_npu_ops.cpp:140

# 查看调用栈
(gdb) bt
(gdb) bt full

# 查看变量
(gdb) print variable
(gdb) info locals
(gdb) info args

# 继续执行
(gdb) continue
(gdb) next
(gdb) step
```

### 关键断点位置

- `echo_npu_add_Tensor` - add 操作入口
- `device_to_cpu` - 设备到 CPU 转换
- `echo_npu_copy_from` - 数据拷贝函数

### 调试 segmentation fault

当程序崩溃时：

```gdb
(gdb) bt              # 查看调用栈
(gdb) frame 0         # 切换到崩溃帧
(gdb) info locals     # 查看局部变量
(gdb) print tensor.is_contiguous()  # 检查 tensor 状态
```

---

## Valgrind 内存调试

### 基本使用

```bash
# 检测内存泄漏
valgrind --leak-check=full python your_script.py

# 完整调试命令
valgrind \
  --leak-check=full \
  --show-leak-kinds=all \
  --track-origins=yes \
  --log-file=valgrind_output.log \
  python your_script.py
```

### 针对 PyTorch 扩展

由于 PyTorch 会产生大量噪音，建议：

```bash
# 使用 Python suppressions
wget https://github.com/python/cpython/raw/main/Misc/valgrind-python.supp
valgrind --suppressions=valgrind-python.supp python your_script.py

# 只关注特定错误
valgrind --leak-check=no python your_script.py 2>&1 | grep "echo_npu\|invalid"
```

### 常见错误类型

- **Invalid read/write**: 访问已释放的内存或数组越界
- **Memory leak**: 分配的内存没有释放
- **Invalid free**: 重复释放或释放未分配的内存
- **Use of uninitialized value**: 使用未初始化的变量

---

## AddressSanitizer

AddressSanitizer 比 Valgrind 更快，适合快速检测内存问题。

### 编译启用

```bash
export CFLAGS="-fsanitize=address -g -O0"
export CXXFLAGS="-fsanitize=address -g -O0"
export LDFLAGS="-fsanitize=address"
python setup.py clean --all
python setup.py build_ext --inplace
```

### 运行测试

```bash
# 禁用 CUDA（避免兼容性问题）
export CUDA_VISIBLE_DEVICES=""
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=1
export LD_PRELOAD=$(gcc -print-file-name=libasan.so)
python test_asan.py 2>&1 | tee asan_output.log
```

### 常见问题

#### 问题 1: ASan 运行时库加载顺序

```
==12345==ASan runtime does not come first in initial library list
```

**解决方案**：
```bash
export LD_PRELOAD=$(gcc -print-file-name=libasan.so)
```

#### 问题 2: 与 PyTorch CUDA 库冲突

如果遇到 CUDA 相关错误，可以：
- 禁用 CUDA: `export CUDA_VISIBLE_DEVICES=""`
- 使用简化的测试程序（不加载完整模型）
- 使用 suppressions 文件

### 预期输出格式

```
==12345==ERROR: AddressSanitizer: heap-use-after-free
READ of size 4 at 0x7f8b8c000000 thread T0
    #0 0x4e8f123 in echoNpuMemcpy simulator.cpp:140
    #1 0x4e8f456 in echo_npu_copy_from echo_npu_ops.cpp:459

SUMMARY: AddressSanitizer: heap-use-after-free simulator.cpp:140
```

---

## 快速参考

### 调试工作流

1. **开发阶段**：使用 AddressSanitizer（快速反馈）
2. **测试阶段**：使用 Valgrind（全面检查）
3. **调试阶段**：使用 GDB（精确定位）

### 工具对比

| 工具 | 速度 | 检测范围 | 需要重新编译 |
|-----|------|---------|------------|
| GDB | 快 | 崩溃调试 | 否（建议使用 -g） |
| Valgrind | 慢（10-50x） | 全面内存检查 | 否 |
| AddressSanitizer | 中等（2-3x） | 内存错误 | 是 |

### 常用命令速查

```bash
# GDB
gdb --args python script.py
(gdb) break function_name
(gdb) run
(gdb) bt

# Valgrind
valgrind --leak-check=full python script.py

# AddressSanitizer
export CFLAGS="-fsanitize=address -g -O0"
python setup.py build_ext --inplace
export ASAN_OPTIONS=detect_leaks=1
python script.py
```

---

**参考资源**：
- [GDB 官方文档](https://sourceware.org/gdb/documentation/)
- [Valgrind 官方文档](https://valgrind.org/docs/manual/manual.html)
- [AddressSanitizer 文档](https://github.com/google/sanitizers/wiki/AddressSanitizer)

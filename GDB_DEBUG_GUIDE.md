# GDB 调试指南

本指南介绍如何使用 GDB 调试 ECHO-NPU 扩展中的 segmentation fault 问题。

## 快速开始

### 1. 构建调试版本

```bash
# 使用调试构建脚本
./debug_with_gdb.sh

# 或者手动构建
python setup_debug.py build_ext --inplace
```

### 2. 使用 GDB 运行测试

```bash
# 方法1：使用便捷脚本（推荐）
./run_gdb.sh test/test_lenet.py

# 方法2：直接使用 GDB
gdb --args python test/test_lenet.py
```

## GDB 常用命令

### 基本命令

```gdb
# 运行程序
(gdb) run

# 设置断点
(gdb) break echo_npu_add_Tensor          # 在函数入口设置断点
(gdb) break echo_npu_ops.cpp:140         # 在特定行设置断点
(gdb) break device_to_cpu                # 在 device_to_cpu 函数设置断点
(gdb) break echo_npu_copy_from            # 在 _copy_from 函数设置断点

# 查看断点
(gdb) info breakpoints

# 删除断点
(gdb) delete 1                          # 删除断点 1

# 继续执行
(gdb) continue                           # 继续执行到下一个断点
(gdb) next                               # 执行下一行（不进入函数）
(gdb) step                               # 执行下一行（进入函数）
(gdb) finish                             # 执行完当前函数

# 查看调用栈
(gdb) bt                                 # 打印调用栈
(gdb) bt full                            # 打印完整调用栈（含局部变量）
(gdb) frame N                            # 切换到第 N 帧
(gdb) up                                 # 向上移动一帧
(gdb) down                               # 向下移动一帧

# 查看变量和内存
(gdb) print variable                     # 打印变量值
(gdb) print *pointer                     # 打印指针指向的值
(gdb) print tensor.is_contiguous()       # 调用成员函数
(gdb) info locals                        # 显示所有局部变量
(gdb) info args                          # 显示函数参数
(gdb) x/10x address                      # 以十六进制显示内存（10 个值）

# 查看源代码
(gdb) list                               # 显示当前源代码
(gdb) list function_name                 # 显示函数源代码
(gdb) list 140                           # 显示第 140 行附近的代码
```

### 针对 segmentation fault 的调试

当程序崩溃时，GDB 会自动停止。使用以下命令查看崩溃信息：

```gdb
# 查看调用栈
(gdb) bt

# 查看完整调用栈（包含局部变量）
(gdb) bt full

# 切换到崩溃的帧
(gdb) frame 0

# 查看局部变量
(gdb) info locals

# 查看函数参数
(gdb) info args

# 查看源代码
(gdb) list

# 检查 tensor 是否 contiguous
(gdb) print self_device.is_contiguous()
(gdb) print other_device.is_contiguous()

# 检查 tensor 的 stride 信息
(gdb) print self_device.strides()
(gdb) print self_device.sizes()
```

## 关键断点位置

### 1. add.Tensor 函数

```gdb
# 在函数入口设置断点
(gdb) break echo_npu_add_Tensor

# 在检查 contiguous 之前设置断点
(gdb) break echo_npu_ops.cpp:140

# 在调用 device_to_cpu 之前设置断点
(gdb) break echo_npu_ops.cpp:124
```

### 2. device_to_cpu 函数

```gdb
# 在函数入口设置断点
(gdb) break device_to_cpu

# 在 TORCH_CHECK 之前设置断点
(gdb) break echo_npu_ops.cpp:43
```

### 3. _copy_from 函数

```gdb
# 在函数入口设置断点
(gdb) break echo_npu_copy_from

# 在检查 contiguous 之前设置断点
(gdb) break echo_npu_ops.cpp:321
```

## 调试示例

### 示例 1：调试 add.Tensor 中的非 contiguous tensor

```gdb
# 启动 GDB
gdb --args python test/test_lenet.py

# 设置断点
(gdb) break echo_npu_add_Tensor
(gdb) break echo_npu_ops.cpp:140

# 运行
(gdb) run

# 当停在 echo_npu_add_Tensor 时
(gdb) print self.is_contiguous()
(gdb) print other.is_contiguous()
(gdb) print self.strides()
(gdb) print other.strides()

# 继续执行到第 140 行
(gdb) continue

# 检查转换后的 tensor
(gdb) print self_contig.is_contiguous()
(gdb) print other_contig.is_contiguous()
```

### 示例 2：调试 device_to_cpu 中的崩溃

```gdb
# 启动 GDB
gdb --args python test/test_lenet.py

# 设置断点
(gdb) break device_to_cpu
(gdb) break echo_npu_ops.cpp:43

# 运行
(gdb) run

# 当停在 device_to_cpu 时
(gdb) print device_tensor.is_contiguous()
(gdb) print device_tensor.strides()
(gdb) print device_tensor.sizes()

# 如果 is_contiguous() 返回 false，这就是问题所在！
```

### 示例 3：查看崩溃时的完整调用栈

```gdb
# 运行程序直到崩溃
(gdb) run

# 当崩溃发生时，GDB 会自动停止
# 查看调用栈
(gdb) bt

# 查看完整调用栈
(gdb) bt full

# 切换到每一帧查看详细信息
(gdb) frame 0
(gdb) info locals
(gdb) info args
(gdb) list

(gdb) frame 1
(gdb) info locals
(gdb) list
```

## 高级技巧

### 1. 条件断点

```gdb
# 只在 tensor 非 contiguous 时停止
(gdb) break echo_npu_ops.cpp:140 if !self_device.is_contiguous()
```

### 2. 监视变量

```gdb
# 监视变量值的变化
(gdb) watch self_device.is_contiguous()
```

### 3. 打印 tensor 详细信息

```gdb
# 打印 tensor 的所有信息
(gdb) print self_device
(gdb) print self_device.sizes()
(gdb) print self_device.strides()
(gdb) print self_device.storage_offset()
(gdb) print self_device.is_contiguous()
```

### 4. 检查内存

```gdb
# 检查 data_ptr() 指向的内存
(gdb) print self_device.data_ptr()
(gdb) x/10x self_device.data_ptr()      # 查看前 10 个 float 值（十六进制）
(gdb) x/10f self_device.data_ptr()      # 查看前 10 个 float 值（浮点数）
```

## 常见问题排查

### 问题 1：找不到调试符号

如果 GDB 显示 "No debugging symbols found"，确保使用 `setup_debug.py` 构建：

```bash
python setup_debug.py build_ext --inplace
```

### 问题 2：无法设置断点

如果无法在函数上设置断点，可能是：
1. 函数名拼写错误
2. 函数被内联了（使用 `-O0` 可以避免）
3. 需要指定文件：`break echo_npu_ops.cpp:function_name`

### 问题 3：Python 调试支持

如果需要调试 Python 代码，可以安装 Python 调试支持：

```bash
# Ubuntu/Debian
sudo apt-get install python3-dbg

# 然后在 GDB 中加载
(gdb) source /usr/share/gdb/auto-load/usr/bin/python3-gdb.py
```

## 参考资源

- [GDB 官方文档](https://sourceware.org/gdb/documentation/)
- [PyTorch C++ 扩展调试](https://pytorch.org/tutorials/advanced/cpp_extension.html#using-your-extension)
- [GDB Python 调试](https://wiki.python.org/moin/DebuggingWithGdb)

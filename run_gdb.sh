#!/bin/bash
# 使用 GDB 运行 Python 脚本并自动设置断点和命令

if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script> [script_args...]"
    echo "Example: $0 test/test_lenet.py"
    exit 1
fi

SCRIPT="$1"
shift
ARGS="$@"

# 创建 GDB 命令文件
GDB_SCRIPT=$(mktemp /tmp/gdb_commands.XXXXXX)
cat > "$GDB_SCRIPT" << 'EOF'
# GDB 初始化命令
set confirm off
set pagination off

# 设置 Python 相关路径（如果需要）
# set environment PYTHONPATH /path/to/your/project

# 加载 Python 调试支持（如果可用）
# source /usr/share/gdb/auto-load/usr/bin/python3-gdb.py

# 设置断点（在关键函数处）
# 可以在运行时使用 'break' 命令添加更多断点

# 运行程序
run

# 如果崩溃，打印调用栈
if $_siginfo
    echo \n
    echo ==========================================
    echo SEGMENTATION FAULT DETECTED!
    echo ==========================================
    echo \n
    echo Backtrace:
    bt
    echo \n
    echo Backtrace (full):
    bt full
    echo \n
    echo Local variables:
    info locals
    echo \n
    echo Registers:
    info registers
    echo \n
    echo ==========================================
    echo You can now inspect the crash:
    echo   - Use 'frame N' to switch to frame N
    echo   - Use 'print variable' to inspect variables
    echo   - Use 'list' to see source code
    echo ==========================================
end

# 保持 GDB 运行以便检查
# 注释掉下面这行以在崩溃后自动退出
# quit
EOF

echo "=========================================="
echo "Running with GDB..."
echo "=========================================="
echo "Script: $SCRIPT"
echo "Args: $ARGS"
echo ""
echo "GDB commands will be executed from: $GDB_SCRIPT"
echo ""
echo "Useful GDB commands:"
echo "  (gdb) break echo_npu_add_Tensor    # 在 add.Tensor 函数设置断点"
echo "  (gdb) break device_to_cpu         # 在 device_to_cpu 函数设置断点"
echo "  (gdb) break echo_npu_copy_from     # 在 _copy_from 函数设置断点"
echo "  (gdb) break echo_npu_ops.cpp:140   # 在特定行设置断点"
echo "  (gdb) run                          # 运行程序"
echo "  (gdb) bt                           # 打印调用栈"
echo "  (gdb) bt full                      # 打印完整调用栈（含局部变量）"
echo "  (gdb) frame N                      # 切换到第 N 帧"
echo "  (gdb) print variable               # 打印变量值"
echo "  (gdb) list                         # 显示源代码"
echo "  (gdb) info locals                  # 显示局部变量"
echo "  (gdb) continue                     # 继续执行"
echo ""

# 运行 GDB
gdb -x "$GDB_SCRIPT" --args python "$SCRIPT" $ARGS

# 清理临时文件
rm -f "$GDB_SCRIPT"

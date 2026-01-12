#!/bin/bash
# 快速调试脚本 - 构建调试版本并运行 GDB

echo "=========================================="
echo "Quick Debug Setup"
echo "=========================================="

# 构建调试版本
echo "Building DEBUG version..."
rm -rf build/ dist/ *.egg-info echo_npu/*.so echo_npu/_echo_npu_C*.so
python setup_debug.py build_ext --inplace

echo ""
echo "=========================================="
echo "Starting GDB..."
echo "=========================================="
echo ""
echo "Useful commands once in GDB:"
echo "  (gdb) break echo_npu_add_Tensor"
echo "  (gdb) break device_to_cpu"
echo "  (gdb) run"
echo "  (gdb) bt          # when crashed"
echo "  (gdb) bt full     # full backtrace"
echo ""

# 运行 GDB
if [ $# -eq 0 ]; then
    gdb -x gdb_commands.txt --args python test/test_lenet.py
else
    gdb -x gdb_commands.txt --args python "$@"
fi

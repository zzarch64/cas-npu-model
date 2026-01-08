#!/bin/bash
# GDB 调试脚本 - 用于调试 segmentation fault

set -e

echo "=========================================="
echo "Building DEBUG version for GDB..."
echo "=========================================="

# 清理旧的构建
echo ""
echo "Step 1: Cleaning old build..."
rm -rf build/ dist/ *.egg-info cas_npu/*.so cas_npu/_cas_npu_C*.so

# 使用调试版本构建
echo ""
echo "Step 2: Building DEBUG extension..."
python setup_debug.py build_ext --inplace

echo ""
echo "=========================================="
echo "Build completed! Now you can use GDB:"
echo "=========================================="
echo ""
echo "Method 1: Run with GDB directly"
echo "  gdb --args python test/test_lenet.py"
echo ""
echo "Method 2: Use the debug script"
echo "  ./run_gdb.sh test/test_lenet.py"
echo ""
echo "Method 3: Attach to running process"
echo "  # In one terminal:"
echo "  python test/test_lenet.py"
echo "  # In another terminal:"
echo "  gdb -p \$(pgrep -f 'python test/test_lenet.py')"
echo ""

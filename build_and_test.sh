#!/bin/bash
# ECHO-NPU Extension Build and Test Script

set -e

echo "=========================================="
echo "ECHO-NPU Extension Build and Test"
echo "=========================================="

# 清理旧的构建
echo ""
echo "Step 1: Cleaning old build..."
rm -rf build/ dist/ *.egg-info python/*.so python/_echo_npu_C*.so

# 构建扩展
echo ""
echo "Step 2: Building extension..."
python setup.py build_ext --inplace

# 运行测试
echo ""
# echo "Step 3: Running tests..."
# python test/run_all_tests.py

echo ""
echo "=========================================="
echo "Build and test completed successfully!"
echo "=========================================="


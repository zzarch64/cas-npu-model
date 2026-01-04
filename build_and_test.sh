#!/bin/bash
# NPU-CAS Extension Build and Test Script

set -e

echo "=========================================="
echo "NPU-CAS Extension Build and Test"
echo "=========================================="

# 清理旧的构建
echo ""
echo "Step 1: Cleaning old build..."
rm -rf build/ dist/ *.egg-info python/*.so python/_cas_npu_C*.so

# 构建扩展
echo ""
echo "Step 2: Building extension..."
python setup.py build_ext --inplace

# 运行测试
echo ""
echo "Step 3: Running tests..."
python test/test_cas_npu.py

echo ""
echo "=========================================="
echo "Build and test completed successfully!"
echo "=========================================="


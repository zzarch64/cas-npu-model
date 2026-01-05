// CAS-NPU Runtime Header - 模拟NPU runtime和驱动
#pragma once

#include <cstddef>
#include <cstdint>

namespace cas_npu {

// 错误码定义
enum CasNpuError {
    CAS_NPU_SUCCESS = 0,
    CAS_NPU_ERROR_INVALID_DEVICE = 1,
    CAS_NPU_ERROR_OUT_OF_MEMORY = 2,
    CAS_NPU_ERROR_INVALID_VALUE = 3,
    CAS_NPU_ERROR_UNKNOWN = 999
};

// 内存拷贝方向（类似 cudaMemcpyKind）
enum CasNpuMemcpyKind {
    CAS_NPU_MEMCPY_HOST_TO_HOST = 0,      // CPU -> CPU
    CAS_NPU_MEMCPY_HOST_TO_DEVICE = 1,    // CPU -> Device (Host to NPU)
    CAS_NPU_MEMCPY_DEVICE_TO_HOST = 2,    // Device -> CPU (NPU to Host)
    CAS_NPU_MEMCPY_DEVICE_TO_DEVICE = 3,  // Device -> Device (NPU to NPU)
    CAS_NPU_MEMCPY_DEFAULT = 4            // 自动检测方向
};

// 设备数量（模拟1个NPU设备）
constexpr int CAS_NPU_DEVICE_COUNT = 1;

// Runtime API - 模拟NPU驱动接口

// 设备管理
CasNpuError casNpuGetDeviceCount(int* count);
CasNpuError casNpuSetDevice(int device);
CasNpuError casNpuGetDevice(int* device);

// 内存管理
CasNpuError casNpuMalloc(void** ptr, size_t size);
CasNpuError casNpuFree(void* ptr);

// 带方向的内存拷贝（类似 cudaMemcpy）
// dst: 目标地址
// src: 源地址
// size: 拷贝字节数
// kind: 拷贝方向
CasNpuError casNpuMemcpy(void* dst, const void* src, size_t size, CasNpuMemcpyKind kind);

CasNpuError casNpuMemset(void* ptr, int value, size_t size);

// 同步
CasNpuError casNpuDeviceSynchronize();

// 计算操作 - 模拟NPU运算

// 加法运算
CasNpuError casNpuAddTensor(
    float* output,
    const float* input1,
    const float* input2,
    size_t num_elements,
    float alpha = 1.0f);

// 矩阵乘法: output = input1 @ input2
// input1: [M, K], input2: [K, N], output: [M, N]
CasNpuError casNpuMatMul(
    float* output,
    const float* input1,
    const float* input2,
    int64_t M,
    int64_t K,
    int64_t N);

// 批量矩阵乘法: output = input1 @ input2
// input1: [B, M, K], input2: [B, K, N], output: [B, M, N]
CasNpuError casNpuBatchMatMul(
    float* output,
    const float* input1,
    const float* input2,
    int64_t B,
    int64_t M,
    int64_t K,
    int64_t N);

// 张量连接: 将两个张量沿着指定维度连接
// input1: 第一个输入张量的数据指针
// input2: 第二个输入张量的数据指针
// output: 输出张量的数据指针
// dim: 连接的维度
// shape1: 第一个输入张量的形状数组
// shape2: 第二个输入张量的形状数组
// output_shape: 输出张量的形状数组
// ndim: 张量的维度数
CasNpuError casNpuCat(
    float* output,
    const float* input1,
    const float* input2,
    int dim,
    const int64_t* shape1,
    const int64_t* shape2,
    const int64_t* output_shape,
    int ndim);

// 量化操作: 将float32 tensor量化为int8
// input: 输入float32数据指针
// output: 输出int8数据指针
// num_elements: 元素数量
// scale: 量化缩放因子
// zero_point: 量化零点
CasNpuError casNpuQuantize(
    int8_t* output,
    const float* input,
    size_t num_elements,
    float scale,
    int8_t zero_point);

// 获取错误描述
const char* casNpuGetErrorString(CasNpuError error);

} // namespace cas_npu


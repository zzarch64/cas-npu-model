// CAS-NPU Runtime Implementation - 模拟NPU runtime和驱动
// 使用CPU模拟NPU设备的行为

#include "../cas_npu_runtime.h"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <unordered_map>
#include <mutex>
#include <vector>

namespace cas_npu {

// 全局状态
static thread_local int current_device = 0;
static std::mutex allocation_mutex;
static std::unordered_map<void*, size_t> allocations;

// 统计信息
static uint64_t total_allocations = 0;
static uint64_t total_frees = 0;
static uint64_t add_operations = 0;
static uint64_t mm_operations = 0;
static uint64_t bmm_operations = 0;

CasNpuError casNpuGetDeviceCount(int* count) {
    if (count == nullptr) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    *count = CAS_NPU_DEVICE_COUNT;
    return CAS_NPU_SUCCESS;
}

CasNpuError casNpuSetDevice(int device) {
    if (device < 0 || device >= CAS_NPU_DEVICE_COUNT) {
        return CAS_NPU_ERROR_INVALID_DEVICE;
    }
    current_device = device;
    return CAS_NPU_SUCCESS;
}

CasNpuError casNpuGetDevice(int* device) {
    if (device == nullptr) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    *device = current_device;
    return CAS_NPU_SUCCESS;
}

CasNpuError casNpuMalloc(void** ptr, size_t size) {
    if (ptr == nullptr) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    if (size == 0) {
        *ptr = nullptr;
        return CAS_NPU_SUCCESS;
    }
    
    // 使用aligned_alloc确保内存对齐（模拟NPU内存对齐要求）
    void* data = nullptr;
    size_t actual_size = size;  // 实际分配的大小
#ifdef _WIN32
    data = _aligned_malloc(size, 64);
    // Windows上_aligned_malloc分配的大小就是请求的size
    actual_size = size;
#else
    // 确保size是64的倍数以满足aligned_alloc要求
    size_t aligned_size = ((size + 63) / 64) * 64;
    data = aligned_alloc(64, aligned_size);
    // Linux上aligned_alloc分配的大小是aligned_size（可能大于size）
    actual_size = aligned_size;
#endif
    
    if (data == nullptr) {
        return CAS_NPU_ERROR_OUT_OF_MEMORY;
    }
    
    // 初始化内存为0，避免未初始化内存包含NaN/Inf值
    // 注意：虽然PyTorch的empty操作不保证初始化，但为了数值稳定性，我们初始化内存
    // 重要：必须初始化整个实际分配的大小，而不仅仅是请求的size
    // 未初始化的尾部内存可能包含垃圾数据，导致NaN
    memset(data, 0, actual_size);
    
    // 跟踪分配（记录实际分配的大小）
    {
        std::lock_guard<std::mutex> lock(allocation_mutex);
        allocations[data] = actual_size;
        total_allocations++;
    }
    
    *ptr = data;
    return CAS_NPU_SUCCESS;
}

CasNpuError casNpuFree(void* ptr) {
    if (ptr == nullptr) {
        return CAS_NPU_SUCCESS;
    }
    
    {
        std::lock_guard<std::mutex> lock(allocation_mutex);
        auto it = allocations.find(ptr);
        if (it == allocations.end()) {
            return CAS_NPU_ERROR_INVALID_VALUE;
        }
        allocations.erase(it);
        total_frees++;
    }
    
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
    
    return CAS_NPU_SUCCESS;
}

// 带方向的内存拷贝（类似 cudaMemcpy）
// 在 CPU 模拟器中，所有方向的拷贝都是简单的 memcpy
// 实际硬件实现中，不同方向可能需要不同的 DMA 操作
CasNpuError casNpuMemcpy(void* dst, const void* src, size_t size, CasNpuMemcpyKind kind) {
    if ((dst == nullptr || src == nullptr) && size > 0) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    
    // 在 CPU 模拟器中，所有内存都在同一地址空间
    // 所以无论 kind 是什么，都使用 memcpy
    // 实际硬件中，Host->Device 和 Device->Host 可能需要 PCIe/DMA 传输
    // Device->Device 可能需要片上内存拷贝
    if (size > 0) {
        switch (kind) {
            case CAS_NPU_MEMCPY_HOST_TO_HOST:
            case CAS_NPU_MEMCPY_HOST_TO_DEVICE:
            case CAS_NPU_MEMCPY_DEVICE_TO_HOST:
            case CAS_NPU_MEMCPY_DEVICE_TO_DEVICE:
            case CAS_NPU_MEMCPY_DEFAULT:
            default:
                memcpy(dst, src, size);
                break;
        }
    }
    return CAS_NPU_SUCCESS;
}

CasNpuError casNpuMemset(void* ptr, int value, size_t size) {
    if (ptr == nullptr && size > 0) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    if (size > 0) {
        memset(ptr, value, size);
    }
    return CAS_NPU_SUCCESS;
}

CasNpuError casNpuDeviceSynchronize() {
    // CPU模拟，无需实际同步
    return CAS_NPU_SUCCESS;
}

// 核心功能：模拟NPU加法运算
CasNpuError casNpuAddTensor(
    float* output,
    const float* input1,
    const float* input2,
    size_t num_elements,
    float alpha) {
    
    if (output == nullptr || input1 == nullptr || input2 == nullptr) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    
    // 模拟NPU加法: output = input1 + alpha * input2
    for (size_t i = 0; i < num_elements; ++i) {
        output[i] = input1[i] + alpha * input2[i];
    }
    
    add_operations++;
    
    return CAS_NPU_SUCCESS;
}

// 矩阵乘法实现: output = input1 @ input2
// 使用简单的三重循环实现 (实际NPU会有优化实现)
CasNpuError casNpuMatMul(
    float* output,
    const float* input1,
    const float* input2,
    int64_t M,
    int64_t K,
    int64_t N) {
    
    if (output == nullptr || input1 == nullptr || input2 == nullptr) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    if (M <= 0 || K <= 0 || N <= 0) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    
    // 初始化输出为0
    for (int64_t i = 0; i < M * N; ++i) {
        output[i] = 0.0f;
    }
    
    // 矩阵乘法: output[i, j] = sum_k(input1[i, k] * input2[k, j])
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += input1[i * K + k] * input2[k * N + j];
            }
            output[i * N + j] = sum;
        }
    }
    
    mm_operations++;
    return CAS_NPU_SUCCESS;
}

// 批量矩阵乘法实现: output[b] = input1[b] @ input2[b]
CasNpuError casNpuBatchMatMul(
    float* output,
    const float* input1,
    const float* input2,
    int64_t B,
    int64_t M,
    int64_t K,
    int64_t N) {
    
    if (output == nullptr || input1 == nullptr || input2 == nullptr) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    if (B <= 0 || M <= 0 || K <= 0 || N <= 0) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    
    // 对每个batch执行矩阵乘法
    for (int64_t b = 0; b < B; ++b) {
        const float* batch_input1 = input1 + b * M * K;
        const float* batch_input2 = input2 + b * K * N;
        float* batch_output = output + b * M * N;
        
        auto err = casNpuMatMul(batch_output, batch_input1, batch_input2, M, K, N);
        if (err != CAS_NPU_SUCCESS) {
            return err;
        }
    }
    
    bmm_operations++;
    return CAS_NPU_SUCCESS;
}

// 张量连接实现: 将两个张量沿着指定维度连接
CasNpuError casNpuCat(
    float* output,
    const float* input1,
    const float* input2,
    int dim,
    const int64_t* shape1,
    const int64_t* shape2,
    const int64_t* output_shape,
    int ndim) {
    
    if (output == nullptr || input1 == nullptr || input2 == nullptr) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    if (shape1 == nullptr || shape2 == nullptr || output_shape == nullptr) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    if (dim < 0 || dim >= ndim) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    
    // 验证形状：除了连接维度，其他维度必须相同
    for (int i = 0; i < ndim; ++i) {
        if (i != dim && shape1[i] != shape2[i]) {
            return CAS_NPU_ERROR_INVALID_VALUE;
        }
    }
    
    // 计算连接维度之前的总元素数（前部分）
    int64_t prefix_size = 1;
    for (int i = 0; i < dim; ++i) {
        prefix_size *= shape1[i];
    }
    
    // 计算连接维度的大小
    int64_t dim1_size = shape1[dim];
    int64_t dim2_size = shape2[dim];
    int64_t output_dim_size = dim1_size + dim2_size;
    
    // 计算连接维度之后的总元素数（后部分）
    int64_t suffix_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        suffix_size *= shape1[i];
    }
    
    // 计算每个输入在连接维度上的步长
    int64_t dim_stride = suffix_size;
    
    // 执行连接操作
    for (int64_t p = 0; p < prefix_size; ++p) {
        // 复制第一个输入的数据
        const float* src1 = input1 + p * dim1_size * dim_stride;
        float* dst = output + p * output_dim_size * dim_stride;
        for (int64_t d = 0; d < dim1_size; ++d) {
            const float* src_row = src1 + d * dim_stride;
            float* dst_row = dst + d * dim_stride;
            memcpy(dst_row, src_row, suffix_size * sizeof(float));
        }
        
        // 复制第二个输入的数据
        const float* src2 = input2 + p * dim2_size * dim_stride;
        dst = output + p * output_dim_size * dim_stride + dim1_size * dim_stride;
        for (int64_t d = 0; d < dim2_size; ++d) {
            const float* src_row = src2 + d * dim_stride;
            float* dst_row = dst + d * dim_stride;
            memcpy(dst_row, src_row, suffix_size * sizeof(float));
        }
    }
    
    return CAS_NPU_SUCCESS;
}

// 量化操作实现: 将float32量化为int8
CasNpuError casNpuQuantize(
    int8_t* output,
    const float* input,
    size_t num_elements,
    float scale,
    int8_t zero_point) {
    
    if (output == nullptr || input == nullptr) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    if (num_elements == 0) {
        return CAS_NPU_SUCCESS;
    }
    if (scale <= 0.0f) {
        return CAS_NPU_ERROR_INVALID_VALUE;
    }
    
    // 量化公式: quantized = round(input / scale) + zero_point
    // 然后clamp到int8范围 [-128, 127]
    for (size_t i = 0; i < num_elements; ++i) {
        float quantized_float = input[i] / scale + static_cast<float>(zero_point);
        int quantized_int = static_cast<int>(std::round(quantized_float));
        
        // Clamp到int8范围
        if (quantized_int > 127) {
            quantized_int = 127;
        } else if (quantized_int < -128) {
            quantized_int = -128;
        }
        
        output[i] = static_cast<int8_t>(quantized_int);
    }
    
    return CAS_NPU_SUCCESS;
}

const char* casNpuGetErrorString(CasNpuError error) {
    switch (error) {
        case CAS_NPU_SUCCESS:
            return "NPU-CAS Success";
        case CAS_NPU_ERROR_INVALID_DEVICE:
            return "NPU-CAS Invalid device";
        case CAS_NPU_ERROR_OUT_OF_MEMORY:
            return "NPU-CAS Out of memory";
        case CAS_NPU_ERROR_INVALID_VALUE:
            return "NPU-CAS Invalid value";
        case CAS_NPU_ERROR_UNKNOWN:
        default:
            return "NPU-CAS Unknown error";
    }
}

} // namespace cas_npu


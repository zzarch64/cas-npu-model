// ECHO-NPU 自定义算子示例
// 演示如何注册PyTorch中不存在的新算子 - 自定义量化算子

#include "runtime/echo_npu_runtime.h"
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>
#include <vector>
#include <cstdint>

using namespace echo_npu;

namespace {

// ============ 步骤1: 定义算子的schema（在自定义命名空间 echo_npu 中）============
// 使用 TORCH_LIBRARY 宏定义新命名空间和新算子的schema
TORCH_LIBRARY(echo_npu, m) {
    // 定义自定义量化算子的schema
    // 参数说明：
    //   - Tensor input: 输入的float32 tensor
    //   - float scale: 量化缩放因子
    //   - int zero_point: 量化零点（int8范围：-128到127）
    // 返回值：
    //   - Tensor: 量化后的int8 tensor
    m.def("custom_quantize(Tensor input, float scale, int zero_point) -> Tensor");
}

// ============ 步骤2: 实现算子函数 ============
// 实现自定义量化算子：将float32 tensor量化为int8 tensor
at::Tensor echo_npu_custom_quantize(
    const at::Tensor& input,
    double scale,
    int64_t zero_point) {
    
    // 参数检查
    TORCH_CHECK(input.device().is_privateuseone(), 
                "Expected input tensor on ECHO-NPU device");
    TORCH_CHECK(input.scalar_type() == at::kFloat,
                "Input tensor must be float32");
    TORCH_CHECK(scale > 0.0, "Scale must be positive");
    TORCH_CHECK(zero_point >= -128 && zero_point <= 127,
                "Zero point must be in range [-128, 127]");
    
    // 创建输出tensor（int8类型，与输入相同的形状和设备）
    // 注意：在PyTorch中，int8类型使用ScalarType::Char表示
    auto output = at::empty(
        input.sizes(),
        input.options().dtype(c10::ScalarType::Char)
    );
    
    // 获取数据指针
    // 确保tensor是contiguous的，然后获取数据指针
    // 由于empty创建的tensor已经是contiguous的，可以直接使用data_ptr()
    // 如果data_ptr<T>()模板不工作，使用static_cast转换
    void* out_ptr = output.data_ptr();
    int8_t* out_data = static_cast<int8_t*>(out_ptr);
    const float* input_data = input.data_ptr<float>();
    size_t num_elements = input.numel();
    
    // 调用ECHO-NPU runtime执行量化操作
    auto err = echoNpuQuantize(
        out_data,
        input_data,
        num_elements,
        static_cast<float>(scale),
        static_cast<int8_t>(zero_point)
    );
    
    TORCH_CHECK(err == ECHO_NPU_SUCCESS,
                "ECHO-NPU quantize operation failed: ", echoNpuGetErrorString(err));
    
    return output;
}

// ============ 步骤3: 为PrivateUse1设备注册实现 ============
// 使用 TORCH_LIBRARY_IMPL 为特定设备（PrivateUse1）注册算子实现
TORCH_LIBRARY_IMPL(echo_npu, PrivateUse1, m) {
    // 将算子名称与实现函数绑定
    m.impl("custom_quantize", &echo_npu_custom_quantize);
}

} // anonymous namespace

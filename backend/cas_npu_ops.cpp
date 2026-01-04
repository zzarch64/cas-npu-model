// CAS-NPU Operations Implementation
// 注册操作到PrivateUse1和AutogradPrivateUse1 dispatch key

#include "runtime/cas_npu_runtime.h"

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/_local_scalar_dense_native.h>
#include <ATen/ops/as_strided_native.h>
#include <torch/library.h>
#include <vector>

using namespace cas_npu;

namespace {

// ============ 全局CPU Fallback ============

void cas_npu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    at::native::cpu_fallback(op, stack);
}

// ============ 核心操作实现 ============

// add.Tensor实现 - 调用CAS-NPU runtime完成加法运算
at::Tensor cas_npu_add_Tensor(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
    
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(other.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(self.sizes() == other.sizes(),
                "Tensor sizes must match for add operation");
    TORCH_CHECK(self.scalar_type() == at::kFloat,
                "Only float tensors supported for now");
    
    // 创建输出tensor
    auto output = at::empty(self.sizes(), self.options());
    
    // 获取数据指针
    float* out_data = output.data_ptr<float>();
    const float* self_data = self.data_ptr<float>();
    const float* other_data = other.data_ptr<float>();
    
    // 调用CAS-NPU runtime执行加法
    auto err = casNpuAddTensor(
        out_data,
        self_data,
        other_data,
        self.numel(),
        alpha.toFloat()
    );
    
    TORCH_CHECK(err == CAS_NPU_SUCCESS,
                "CAS-NPU add operation failed: ", casNpuGetErrorString(err));
    
    return output;
}

// mm实现 - 矩阵乘法: output = input @ mat2
at::Tensor cas_npu_mm(
    const at::Tensor& input,
    const at::Tensor& mat2) {
    
    TORCH_CHECK(input.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(mat2.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(input.dim() == 2, "input must be 2D tensor");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be 2D tensor");
    TORCH_CHECK(input.size(1) == mat2.size(0),
                "input size(1) must match mat2 size(0)");
    TORCH_CHECK(input.scalar_type() == at::kFloat,
                "Only float tensors supported for now");
    TORCH_CHECK(mat2.scalar_type() == at::kFloat,
                "Only float tensors supported for now");
    
    // 创建输出tensor: [M, N] where M = input.size(0), N = mat2.size(1)
    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = mat2.size(1);
    auto output = at::empty({M, N}, input.options());
    
    // 获取数据指针
    float* out_data = output.data_ptr<float>();
    const float* input_data = input.data_ptr<float>();
    const float* mat2_data = mat2.data_ptr<float>();
    
    // 调用CAS-NPU runtime执行矩阵乘法
    auto err = casNpuMatMul(out_data, input_data, mat2_data, M, K, N);
    
    TORCH_CHECK(err == CAS_NPU_SUCCESS,
                "CAS-NPU mm operation failed: ", casNpuGetErrorString(err));
    
    return output;
}

// bmm实现 - 批量矩阵乘法: output = input @ mat2
at::Tensor cas_npu_bmm(
    const at::Tensor& input,
    const at::Tensor& mat2) {
    
    TORCH_CHECK(input.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(mat2.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(input.dim() == 3, "input must be 3D tensor");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3D tensor");
    TORCH_CHECK(input.size(0) == mat2.size(0),
                "batch sizes must match");
    TORCH_CHECK(input.size(2) == mat2.size(1),
                "input size(2) must match mat2 size(1)");
    TORCH_CHECK(input.scalar_type() == at::kFloat,
                "Only float tensors supported for now");
    TORCH_CHECK(mat2.scalar_type() == at::kFloat,
                "Only float tensors supported for now");
    
    // 创建输出tensor: [B, M, N]
    int64_t B = input.size(0);
    int64_t M = input.size(1);
    int64_t K = input.size(2);
    int64_t N = mat2.size(2);
    auto output = at::empty({B, M, N}, input.options());
    
    // 获取数据指针
    float* out_data = output.data_ptr<float>();
    const float* input_data = input.data_ptr<float>();
    const float* mat2_data = mat2.data_ptr<float>();
    
    // 调用CAS-NPU runtime执行批量矩阵乘法
    auto err = casNpuBatchMatMul(out_data, input_data, mat2_data, B, M, K, N);
    
    TORCH_CHECK(err == CAS_NPU_SUCCESS,
                "CAS-NPU bmm operation failed: ", casNpuGetErrorString(err));
    
    return output;
}

// cat实现 - 张量连接: 将多个张量沿着指定维度连接
at::Tensor cas_npu_cat(
    const c10::IListRef<at::Tensor>& tensors,
    int64_t dim) {
    
    if (tensors.empty()) {
        TORCH_CHECK(false, "cat expects at least one tensor");
    }
    
    // 转换为vector以便访问
    std::vector<at::Tensor> tensors_vec;
    tensors_vec.reserve(tensors.size());
    for (const auto& t : tensors) {
        tensors_vec.push_back(t);
    }
    
    // 检查所有张量都在CAS-NPU设备上
    for (const auto& t : tensors_vec) {
        TORCH_CHECK(t.device().is_privateuseone(), 
                    "Expected all tensors on CAS-NPU device");
        TORCH_CHECK(t.scalar_type() == at::kFloat,
                    "Only float tensors supported for now");
    }
    
    // 规范化维度
    int64_t ndim = tensors_vec[0].dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
                "cat: dimension out of range");
    
    // 验证所有张量的形状（除了连接维度）
    auto first_shape = tensors_vec[0].sizes();
    for (size_t i = 1; i < tensors_vec.size(); ++i) {
        auto shape = tensors_vec[i].sizes();
        TORCH_CHECK(shape.size() == first_shape.size(),
                    "cat: all tensors must have the same number of dimensions");
        for (int64_t d = 0; d < ndim; ++d) {
            if (d != dim) {
                TORCH_CHECK(shape[d] == first_shape[d],
                            "cat: Sizes of tensors must match except in dimension ", dim);
            }
        }
    }
    
    // 计算输出形状
    std::vector<int64_t> output_shape(first_shape.begin(), first_shape.end());
    output_shape[dim] = 0;
    for (const auto& t : tensors_vec) {
        output_shape[dim] += t.size(dim);
    }
    
    // 创建输出tensor
    auto output = at::empty(output_shape, tensors_vec[0].options());
    
    // 如果只有一个张量，直接复制
    if (tensors_vec.size() == 1) {
        output.copy_(tensors_vec[0]);
        return output;
    }
    
    // 对于多个张量，使用递归方式两两连接
    // 先连接前两个张量到临时结果
    std::vector<int64_t> current_shape(first_shape.begin(), first_shape.end());
    current_shape[dim] = tensors_vec[0].size(dim) + tensors_vec[1].size(dim);
    auto current_result = at::empty(current_shape, tensors_vec[0].options());
    
    std::vector<int64_t> shape1_vec(first_shape.begin(), first_shape.end());
    std::vector<int64_t> shape2_vec(tensors_vec[1].sizes().begin(), tensors_vec[1].sizes().end());
    
    auto err = casNpuCat(
        current_result.data_ptr<float>(),
        tensors_vec[0].data_ptr<float>(),
        tensors_vec[1].data_ptr<float>(),
        static_cast<int>(dim),
        shape1_vec.data(),
        shape2_vec.data(),
        current_shape.data(),
        static_cast<int>(ndim)
    );
    
    TORCH_CHECK(err == CAS_NPU_SUCCESS,
                "CAS-NPU cat operation failed: ", casNpuGetErrorString(err));
    
    // 继续连接剩余的张量
    for (size_t i = 2; i < tensors_vec.size(); ++i) {
        // 计算新的输出形状
        std::vector<int64_t> next_shape(current_shape.begin(), current_shape.end());
        next_shape[dim] += tensors_vec[i].size(dim);
        auto next_result = at::empty(next_shape, tensors_vec[0].options());
        
        std::vector<int64_t> current_shape_vec(current_shape.begin(), current_shape.end());
        std::vector<int64_t> next_tensor_shape_vec(tensors_vec[i].sizes().begin(), tensors_vec[i].sizes().end());
        
        err = casNpuCat(
            next_result.data_ptr<float>(),
            current_result.data_ptr<float>(),
            tensors_vec[i].data_ptr<float>(),
            static_cast<int>(dim),
            current_shape_vec.data(),
            next_tensor_shape_vec.data(),
            next_shape.data(),
            static_cast<int>(ndim)
        );
        
        TORCH_CHECK(err == CAS_NPU_SUCCESS,
                    "CAS-NPU cat operation failed: ", casNpuGetErrorString(err));
        
        current_result = next_result;
        current_shape = next_shape;
    }
    
    // 将最终结果复制到输出
    output.copy_(current_result);
    
    return output;
}

// cat.out实现 - 带输出参数的cat
at::Tensor& cas_npu_cat_out(
    const c10::IListRef<at::Tensor>& tensors,
    int64_t dim,
    at::Tensor& out) {
    
    auto result = cas_npu_cat(tensors, dim);
    out.copy_(result);
    return out;
}

// ============ 基础操作实现 ============

// empty.memory_format实现
at::Tensor cas_npu_empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
    
    const auto device = c10::device_or_default(device_opt);
    const auto dtype = c10::dtype_or_default(dtype_opt);
    
    TORCH_CHECK(device.is_privateuseone(), 
                "Expected CAS-NPU device, got ", device);
    TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
                "Non strided layout not supported");
    TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt),
                "Pin memory can only be on CPU");
    
    const c10::DeviceGuard device_guard(device);
    constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
    auto allocator = at::GetAllocator(at::kPrivateUse1);
    
    return at::detail::empty_generic(
        size, allocator, pu1_dks, dtype, memory_format_opt);
}

// empty_strided实现
at::Tensor cas_npu_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
    
    const auto device = c10::device_or_default(device_opt);
    const auto dtype = c10::dtype_or_default(dtype_opt);
    
    TORCH_CHECK(device.is_privateuseone());
    TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided);
    TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt));
    
    const c10::DeviceGuard device_guard(device);
    constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
    auto allocator = at::GetAllocator(at::kPrivateUse1);
    
    return at::detail::empty_strided_generic(
        size, stride, allocator, pu1_dks, dtype);
}

// _copy_from实现 - 设备间数据拷贝 (关键函数!)
at::Tensor cas_npu_copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
    
    TORCH_CHECK(self.defined(), "Source tensor is not defined");
    TORCH_CHECK(dst.defined(), "Destination tensor is not defined");
    
    // 同设备拷贝
    if (self.device() == dst.device()) {
        // 将我们设备上的内存视为CPU tensor进行拷贝
        at::Tensor dst_as_cpu = at::from_blob(
            dst.data_ptr(),
            dst.sizes(),
            dst.strides(),
            dst.options().device(at::kCPU));
        const at::Tensor self_as_cpu = at::from_blob(
            self.data_ptr(),
            self.sizes(),
            self.strides(),
            self.options().device(at::kCPU));
        
        at::native::copy_(
            const_cast<at::Tensor&>(dst_as_cpu), self_as_cpu, non_blocking);
    } 
    // CPU到CAS-NPU
    else if (self.is_cpu()) {
        at::Tensor dst_as_cpu = at::from_blob(
            dst.data_ptr(),
            dst.sizes(),
            dst.strides(),
            dst.options().device(at::kCPU));
        
        at::native::copy_(
            const_cast<at::Tensor&>(dst_as_cpu), self, non_blocking);
    }
    // CAS-NPU到CPU
    else {
        at::Tensor self_as_cpu = at::from_blob(
            self.data_ptr(),
            self.sizes(),
            self.strides(),
            self.options().device(at::kCPU));
        
        at::native::copy_(
            const_cast<at::Tensor&>(dst), self_as_cpu, non_blocking);
    }
    
    return dst;
}

// _copy_from_and_resize实现
at::Tensor cas_npu_copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
    
    at::native::resize_(dst, self.sizes(), std::nullopt);
    return at::native::copy_(const_cast<at::Tensor&>(dst), self, false);
}

// view实现
at::Tensor cas_npu_view(const at::Tensor& self, c10::SymIntArrayRef size) {
    return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

// resize_实现
const at::Tensor& cas_npu_resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    std::optional<at::MemoryFormat> memory_format) {
    
    return at::native::resize_(
        self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

// _reshape_alias实现
at::Tensor cas_npu_reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
    return at::native::_reshape_alias(
        self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

// set_.source_Storage实现
at::Tensor& cas_npu_set_source_Storage(at::Tensor& self, at::Storage source) {
    return at::native::set_(self, source);
}

// _local_scalar_dense实现
at::Scalar cas_npu_local_scalar_dense(const at::Tensor& self) {
    // 将设备tensor视为CPU tensor来读取标量值
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    return at::native::_local_scalar_dense_cpu(self_as_cpu);
}

// as_strided实现 - 创建strided view
at::Tensor cas_npu_as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset_opt) {
    
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    int64_t storage_offset = storage_offset_opt.has_value() 
        ? storage_offset_opt.value().expect_int() 
        : self.storage_offset();
    
    // 使用 _reshape_alias 来创建 strided view
    // 注意：_reshape_alias 可能不支持 storage_offset，但对于大多数情况应该可以工作
    // 如果 storage_offset 不同，我们需要特殊处理（这种情况较少见）
    auto result = at::native::_reshape_alias(
        self,
        C10_AS_INTARRAYREF_SLOW(size),
        C10_AS_INTARRAYREF_SLOW(stride)
    );
    
    // 如果 storage_offset 不同，我们需要调整它
    // 但 _reshape_alias 不支持直接设置 storage_offset
    // 对于大多数情况，storage_offset 应该与原始 tensor 相同
    // 如果不同，我们可能需要创建一个新的 tensor view（这种情况较少见）
    if (storage_offset != self.storage_offset()) {
        // 这种情况较少见，暂时使用 _reshape_alias 的结果
        // 如果需要支持不同的 storage_offset，可能需要更复杂的实现
        TORCH_WARN("as_strided with different storage_offset may not work correctly");
    }
    
    return result;
}

// detach实现 - 返回一个detached的view
at::Tensor cas_npu_detach(const at::Tensor& self) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    // detach 返回一个共享相同存储但不参与梯度计算的tensor
    // 创建一个共享存储的 view，使用 as_strided 来保持相同的存储
    auto sizes = self.sizes();
    auto strides = self.strides();
    std::vector<c10::SymInt> sym_sizes(sizes.begin(), sizes.end());
    std::vector<c10::SymInt> sym_strides(strides.begin(), strides.end());
    auto result = cas_npu_as_strided(
        self,
        c10::SymIntArrayRef(sym_sizes),
        c10::SymIntArrayRef(sym_strides),
        c10::SymInt(self.storage_offset())
    );
    result.set_requires_grad(false);
    return result;
}

// t实现 - 2D tensor转置
at::Tensor cas_npu_t(const at::Tensor& self) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(self.dim() == 2, "t() expects a 2D tensor");
    
    // 转置操作：交换维度0和1，这是一个view操作
    // 使用 as_strided 来创建转置的view，交换size和stride
    int64_t M = self.size(0);
    int64_t N = self.size(1);
    auto strides = self.strides();
    
    // 交换size和stride来创建转置的view
    std::vector<c10::SymInt> new_size = {c10::SymInt(N), c10::SymInt(M)};
    std::vector<c10::SymInt> new_stride = {c10::SymInt(strides[1]), c10::SymInt(strides[0])};
    
    return cas_npu_as_strided(
        self,
        c10::SymIntArrayRef(new_size),
        c10::SymIntArrayRef(new_stride),
        c10::SymInt(self.storage_offset())
    );
}

// ============ 操作注册 ============

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // ========== 自定义实现的操作 ==========
    m.impl("add.Tensor", &cas_npu_add_Tensor);
    m.impl("mm", &cas_npu_mm);
    m.impl("bmm", &cas_npu_bmm);
    m.impl("cat", &cas_npu_cat);
    m.impl("cat.out", &cas_npu_cat_out);
    
    // 基础tensor操作
    m.impl("empty.memory_format", &cas_npu_empty_memory_format);
    m.impl("empty_strided", &cas_npu_empty_strided);
    m.impl("_copy_from", &cas_npu_copy_from);
    m.impl("_copy_from_and_resize", &cas_npu_copy_from_and_resize);
    m.impl("view", &cas_npu_view);
    m.impl("resize_", &cas_npu_resize_);
    m.impl("_reshape_alias", &cas_npu_reshape_alias);
    m.impl("set_.source_Storage", &cas_npu_set_source_Storage);
    m.impl("_local_scalar_dense", &cas_npu_local_scalar_dense);
    m.impl("as_strided", &cas_npu_as_strided);
    m.impl("detach", &cas_npu_detach);
    m.impl("t", &cas_npu_t);
    
    // ========== 使用CPU Fallback的操作 ==========
    
    // 基础数学操作
    m.impl("mul.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mul.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sub.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("div.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("neg", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("pow.Tensor_Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // Tensor工厂和操作
    m.impl("ones_like", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("zeros_like", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("full_like", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("fill_.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("zero_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    // arange 操作 - 创建序列
    m.impl("arange.start_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("arange.start_step", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("arange", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 规约操作
    m.impl("sum", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sum.dim_IntList", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mean", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mean.dim", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // Tensor操作
    m.impl("clone", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("contiguous", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("expand", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    // as_strided 已在上面实现，移除fallback注册
    // m.impl("as_strided", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("squeeze", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("squeeze.dim", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("unsqueeze", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    // as_strided, t 和 detach 已在上面实现，移除fallback注册
    // m.impl("as_strided", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    // m.impl("t", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    // 注意：copy_不能使用CPU fallback，会导致递归调用
    // m.impl("copy_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    // m.impl("detach", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("slice.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("select.int", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("index_select", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());  // Embedding需要
    m.impl("embedding", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());  // 直接注册embedding
    
    // 卷积和池化
    m.impl("convolution_overrideable", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("convolution_backward_overrideable", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("max_pool2d_with_indices", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("max_pool2d_with_indices_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 激活函数
    m.impl("relu", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("threshold_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sigmoid", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("tanh", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 三角函数和其他数学函数
    m.impl("cos", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("cos.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sin", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sin.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 矩阵运算
    // mm 和 bmm 已在上面实现，这里移除fallback注册
    // m.impl("mm", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("addmm", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    // m.impl("bmm", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // Loss相关
    m.impl("log_softmax.int", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_log_softmax_backward_data", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("nll_loss_forward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("nll_loss_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mse_loss", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mse_loss_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 打印相关
    m.impl("abs.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isnan", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isinf", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isfinite", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("ne.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("eq.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // SGD优化器相关
    m.impl("add.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("add_.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mul_.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("addcmul_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("addcdiv_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sqrt_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
}

// ============ AutogradPrivateUse1 注册 ============

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}

} // anonymous namespace

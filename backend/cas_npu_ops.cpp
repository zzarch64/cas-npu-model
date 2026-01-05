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
    
    // 至少有一个 tensor 必须在 CAS-NPU 上
    bool self_on_device = self.device().is_privateuseone();
    bool other_on_device = other.device().is_privateuseone();
    TORCH_CHECK(self_on_device || other_on_device, 
                "Expected at least one tensor on CAS-NPU device");
    
    // 处理混合设备情况（如 tensor + scalar_tensor）
    // 如果 other 是 CPU 上的标量 tensor
    if (self_on_device && !other_on_device && other.numel() == 1) {
        float scalar_val = other.item<float>() * alpha.toFloat();
        auto output = at::empty(self.sizes(), self.options());
        
        float* out_data = output.data_ptr<float>();
        const float* self_data = self.data_ptr<float>();
        int64_t numel = self.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_data[i] = self_data[i] + scalar_val;
        }
        return output;
    }
    
    // 如果 self 是 CPU 上的标量，other 在设备上
    if (!self_on_device && other_on_device && self.numel() == 1) {
        float scalar_val = self.item<float>();
        float alpha_val = alpha.toFloat();
        auto output = at::empty(other.sizes(), other.options());
        
        float* out_data = output.data_ptr<float>();
        const float* other_data = other.data_ptr<float>();
        int64_t numel = other.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_data[i] = scalar_val + other_data[i] * alpha_val;
        }
        return output;
    }
    
    // 两个 tensor 都在 CAS-NPU 上
    TORCH_CHECK(self_on_device && other_on_device,
                "Both tensors must be on CAS-NPU device for element-wise add");
    TORCH_CHECK(self.scalar_type() == at::kFloat,
                "Only float tensors supported for now");
    
    // 处理广播
    auto output_size = at::infer_size(self.sizes(), other.sizes());
    auto output = at::empty(output_size, self.options());
    
    // 如果需要广播，使用 from_blob 方式
    if (self.sizes() != other.sizes()) {
        at::Tensor self_as_cpu = at::from_blob(
            self.data_ptr(),
            self.sizes(),
            self.strides(),
            self.options().device(at::kCPU));
        at::Tensor other_as_cpu = at::from_blob(
            other.data_ptr(),
            other.sizes(),
            other.strides(),
            other.options().device(at::kCPU));
        at::Tensor out_as_cpu = at::from_blob(
            output.data_ptr(),
            output.sizes(),
            output.strides(),
            output.options().device(at::kCPU));
        
        at::add_out(out_as_cpu, self_as_cpu, other_as_cpu, alpha);
        return output;
    }
    
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
    at::Tensor reference_tensor;  // 用于确定输出形状的参考tensor
    int64_t max_ndim = 0;
    
    for (const auto& t : tensors) {
        TORCH_CHECK(t.device().is_privateuseone(), 
                    "Expected all tensors on CAS-NPU device");
        TORCH_CHECK(t.scalar_type() == at::kFloat,
                    "Only float tensors supported for now");
        
        tensors_vec.push_back(t);
        
        // 找到维度最高的非空tensor作为参考
        if (t.numel() > 0 && t.dim() > max_ndim) {
            max_ndim = t.dim();
            reference_tensor = t;
        }
    }
    
    // 如果所有tensor都为空，返回第一个tensor的空副本
    if (!reference_tensor.defined()) {
        return tensors_vec[0].clone();
    }
    
    // 规范化维度
    int64_t ndim = reference_tensor.dim();
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
                "cat: dimension out of range");
    
    // 过滤出有效的tensor（numel > 0 且维度匹配）
    std::vector<at::Tensor> valid_tensors;
    auto ref_shape = reference_tensor.sizes();
    
    for (const auto& t : tensors_vec) {
        // 跳过空tensor或维度不匹配的tensor
        if (t.numel() == 0) continue;
        if (t.dim() != ndim) continue;
        
        // 检查除了 cat 维度外的其他维度是否匹配
        bool shape_match = true;
        auto shape = t.sizes();
        for (int64_t d = 0; d < ndim && shape_match; ++d) {
            if (d != dim && shape[d] != ref_shape[d]) {
                shape_match = false;
            }
        }
        
        if (shape_match && t.size(dim) > 0) {
            valid_tensors.push_back(t);
        }
    }
    
    // 如果过滤后没有有效tensor，返回正确形状的空tensor
    if (valid_tensors.empty()) {
        std::vector<int64_t> output_shape(ref_shape.begin(), ref_shape.end());
        output_shape[dim] = 0;
        return at::empty(output_shape, reference_tensor.options());
    }
    
    // 如果只有一个有效tensor
    if (valid_tensors.size() == 1) {
        return valid_tensors[0].clone();
    }
    
    // 计算输出形状
    std::vector<int64_t> output_shape(ref_shape.begin(), ref_shape.end());
    output_shape[dim] = 0;
    for (const auto& t : valid_tensors) {
        output_shape[dim] += t.size(dim);
    }
    
    // 创建输出tensor
    auto output = at::empty(output_shape, reference_tensor.options());
    
    // 使用 from_blob + CPU cat 来实现（更简单可靠）
    std::vector<at::Tensor> cpu_tensors;
    for (const auto& t : valid_tensors) {
        at::Tensor t_as_cpu = at::from_blob(
            t.data_ptr(),
            t.sizes(),
            t.strides(),
            t.options().device(at::kCPU));
        cpu_tensors.push_back(t_as_cpu);
    }
    
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    
    // 使用 CPU cat
    at::cat_out(out_as_cpu, cpu_tensors, dim);
    
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
    
    // 使用 native as_strided 来创建正确的 strided view
    return at::native::as_strided_tensorimpl(
        self,
        C10_AS_INTARRAYREF_SLOW(size),
        C10_AS_INTARRAYREF_SLOW(stride),
        storage_offset);
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

// unsqueeze实现 - 在指定维度插入大小为1的维度（视图操作）
at::Tensor cas_npu_unsqueeze(const at::Tensor& self, int64_t dim) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    int64_t ndim = self.dim();
    // 规范化负数索引
    if (dim < 0) {
        dim = dim + ndim + 1;
    }
    TORCH_CHECK(dim >= 0 && dim <= ndim,
                "unsqueeze: dimension out of range");
    
    // 计算新的shape和stride
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    new_sizes.reserve(ndim + 1);
    new_strides.reserve(ndim + 1);
    
    for (int64_t i = 0; i < ndim + 1; ++i) {
        if (i == dim) {
            new_sizes.push_back(c10::SymInt(1));
            // 对于插入的维度，stride可以是任意值（因为size=1）
            // 通常设置为相邻维度的stride
            if (i < ndim) {
                new_strides.push_back(c10::SymInt(old_strides[i]));
            } else {
                // 最后一个维度
                new_strides.push_back(c10::SymInt(1));
            }
        } else {
            int64_t old_idx = (i < dim) ? i : i - 1;
            new_sizes.push_back(c10::SymInt(old_sizes[old_idx]));
            new_strides.push_back(c10::SymInt(old_strides[old_idx]));
        }
    }
    
    return cas_npu_as_strided(
        self,
        c10::SymIntArrayRef(new_sizes),
        c10::SymIntArrayRef(new_strides),
        c10::SymInt(self.storage_offset())
    );
}

// squeeze.dim实现 - 移除指定维度（如果大小为1）
at::Tensor cas_npu_squeeze_dim(const at::Tensor& self, int64_t dim) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    int64_t ndim = self.dim();
    // 规范化负数索引
    if (dim < 0) {
        dim = dim + ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim,
                "squeeze: dimension out of range");
    
    // 如果该维度大小不是1，返回原tensor
    if (self.size(dim) != 1) {
        return self;
    }
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    new_sizes.reserve(ndim - 1);
    new_strides.reserve(ndim - 1);
    
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            new_sizes.push_back(c10::SymInt(old_sizes[i]));
            new_strides.push_back(c10::SymInt(old_strides[i]));
        }
    }
    
    return cas_npu_as_strided(
        self,
        c10::SymIntArrayRef(new_sizes),
        c10::SymIntArrayRef(new_strides),
        c10::SymInt(self.storage_offset())
    );
}

// squeeze实现 - 移除所有大小为1的维度
at::Tensor cas_npu_squeeze(const at::Tensor& self) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    int64_t ndim = self.dim();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    
    for (int64_t i = 0; i < ndim; ++i) {
        if (old_sizes[i] != 1) {
            new_sizes.push_back(c10::SymInt(old_sizes[i]));
            new_strides.push_back(c10::SymInt(old_strides[i]));
        }
    }
    
    // 如果没有维度被移除
    if (new_sizes.size() == static_cast<size_t>(ndim)) {
        return self;
    }
    
    return cas_npu_as_strided(
        self,
        c10::SymIntArrayRef(new_sizes),
        c10::SymIntArrayRef(new_strides),
        c10::SymInt(self.storage_offset())
    );
}

// expand实现 - 扩展tensor到更大的size（视图操作，通过设置stride=0实现广播）
at::Tensor cas_npu_expand(
    const at::Tensor& self, 
    c10::SymIntArrayRef size,
    bool implicit) {
    
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    int64_t ndim = size.size();
    int64_t self_ndim = self.dim();
    
    TORCH_CHECK(ndim >= self_ndim,
                "expand: the number of sizes provided (", ndim, 
                ") must be greater or equal to the number of dimensions in the tensor (", self_ndim, ")");
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    new_sizes.reserve(ndim);
    new_strides.reserve(ndim);
    
    // 从右向左对齐维度
    int64_t offset = ndim - self_ndim;
    
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t size_val = size[i].expect_int();
        
        if (i < offset) {
            // 新增的维度（在前面）
            TORCH_CHECK(size_val >= 0, "expand: size must be non-negative");
            new_sizes.push_back(size[i]);
            new_strides.push_back(c10::SymInt(0));  // 广播维度stride=0
        } else {
            int64_t old_idx = i - offset;
            int64_t old_size = old_sizes[old_idx];
            int64_t old_stride = old_strides[old_idx];
            
            if (size_val == -1) {
                // -1表示保持原大小
                new_sizes.push_back(c10::SymInt(old_size));
                new_strides.push_back(c10::SymInt(old_stride));
            } else if (old_size == 1) {
                // 原大小为1，可以扩展
                new_sizes.push_back(size[i]);
                new_strides.push_back(c10::SymInt(0));  // 广播维度stride=0
            } else if (old_size == size_val) {
                // 大小相同，保持不变
                new_sizes.push_back(c10::SymInt(old_size));
                new_strides.push_back(c10::SymInt(old_stride));
            } else {
                TORCH_CHECK(false, "expand: the expanded size of the tensor (", size_val,
                           ") must match the existing size (", old_size, ") at non-singleton dimension ", i);
            }
        }
    }
    
    return cas_npu_as_strided(
        self,
        c10::SymIntArrayRef(new_sizes),
        c10::SymIntArrayRef(new_strides),
        c10::SymInt(self.storage_offset())
    );
}

// slice.Tensor实现 - 切片操作（视图操作）
at::Tensor cas_npu_slice_Tensor(
    const at::Tensor& self,
    int64_t dim,
    c10::optional<c10::SymInt> start_opt,
    c10::optional<c10::SymInt> end_opt,
    c10::SymInt step) {
    
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    int64_t ndim = self.dim();
    // 规范化维度
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "slice: dimension out of range");
    
    int64_t dim_size = self.size(dim);
    int64_t step_val = step.expect_int();
    TORCH_CHECK(step_val > 0, "slice step must be positive");
    
    // 处理start
    int64_t start_val = start_opt.has_value() ? start_opt.value().expect_int() : 0;
    if (start_val < 0) {
        start_val += dim_size;
    }
    start_val = std::max(int64_t(0), std::min(start_val, dim_size));
    
    // 处理end
    int64_t end_val = end_opt.has_value() ? end_opt.value().expect_int() : dim_size;
    if (end_val < 0) {
        end_val += dim_size;
    }
    end_val = std::max(int64_t(0), std::min(end_val, dim_size));
    
    // 计算切片后的大小
    int64_t slice_size = (end_val - start_val + step_val - 1) / step_val;
    if (end_val <= start_val) {
        slice_size = 0;
    }
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    new_sizes.reserve(ndim);
    new_strides.reserve(ndim);
    
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            new_sizes.push_back(c10::SymInt(slice_size));
            new_strides.push_back(c10::SymInt(old_strides[i] * step_val));
        } else {
            new_sizes.push_back(c10::SymInt(old_sizes[i]));
            new_strides.push_back(c10::SymInt(old_strides[i]));
        }
    }
    
    // 计算新的storage_offset
    int64_t new_offset = self.storage_offset() + start_val * old_strides[dim];
    
    return cas_npu_as_strided(
        self,
        c10::SymIntArrayRef(new_sizes),
        c10::SymIntArrayRef(new_strides),
        c10::SymInt(new_offset)
    );
}

// select.int实现 - 选择指定维度的一个索引（降维操作）
at::Tensor cas_npu_select_int(const at::Tensor& self, int64_t dim, int64_t index) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    int64_t ndim = self.dim();
    TORCH_CHECK(ndim > 0, "select: cannot select on a 0-d tensor");
    
    // 规范化维度
    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "select: dimension out of range");
    
    int64_t dim_size = self.size(dim);
    // 规范化索引
    if (index < 0) {
        index += dim_size;
    }
    TORCH_CHECK(index >= 0 && index < dim_size, 
                "select: index ", index, " out of range for dimension ", dim, " with size ", dim_size);
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    new_sizes.reserve(ndim - 1);
    new_strides.reserve(ndim - 1);
    
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            new_sizes.push_back(c10::SymInt(old_sizes[i]));
            new_strides.push_back(c10::SymInt(old_strides[i]));
        }
    }
    
    // 计算新的storage_offset
    int64_t new_offset = self.storage_offset() + index * old_strides[dim];
    
    return cas_npu_as_strided(
        self,
        c10::SymIntArrayRef(new_sizes),
        c10::SymIntArrayRef(new_strides),
        c10::SymInt(new_offset)
    );
}

// transpose.int实现 - 交换两个维度（视图操作）
at::Tensor cas_npu_transpose_int(const at::Tensor& self, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    int64_t ndim = self.dim();
    
    // 规范化维度
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    
    TORCH_CHECK(dim0 >= 0 && dim0 < ndim, "transpose: dimension out of range");
    TORCH_CHECK(dim1 >= 0 && dim1 < ndim, "transpose: dimension out of range");
    
    if (dim0 == dim1) {
        return self;
    }
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    new_sizes.reserve(ndim);
    new_strides.reserve(ndim);
    
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t src_dim = i;
        if (i == dim0) src_dim = dim1;
        else if (i == dim1) src_dim = dim0;
        
        new_sizes.push_back(c10::SymInt(old_sizes[src_dim]));
        new_strides.push_back(c10::SymInt(old_strides[src_dim]));
    }
    
    return cas_npu_as_strided(
        self,
        c10::SymIntArrayRef(new_sizes),
        c10::SymIntArrayRef(new_strides),
        c10::SymInt(self.storage_offset())
    );
}

// permute实现 - 按指定顺序重排维度（视图操作）
at::Tensor cas_npu_permute(const at::Tensor& self, c10::IntArrayRef dims) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    int64_t ndim = self.dim();
    TORCH_CHECK(static_cast<int64_t>(dims.size()) == ndim,
                "permute: number of dims must match tensor dimensions");
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<bool> seen(ndim, false);
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    new_sizes.reserve(ndim);
    new_strides.reserve(ndim);
    
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t d = dims[i];
        if (d < 0) d += ndim;
        TORCH_CHECK(d >= 0 && d < ndim, "permute: dimension out of range");
        TORCH_CHECK(!seen[d], "permute: duplicate dims are not allowed");
        seen[d] = true;
        
        new_sizes.push_back(c10::SymInt(old_sizes[d]));
        new_strides.push_back(c10::SymInt(old_strides[d]));
    }
    
    return cas_npu_as_strided(
        self,
        c10::SymIntArrayRef(new_sizes),
        c10::SymIntArrayRef(new_strides),
        c10::SymInt(self.storage_offset())
    );
}

// reshape实现 - 改变tensor形状（可能是view或copy）
at::Tensor cas_npu_reshape(const at::Tensor& self, c10::SymIntArrayRef shape) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    // 计算总元素数并处理 -1
    int64_t total_elements = self.numel();
    std::vector<int64_t> new_shape;
    int64_t infer_idx = -1;
    int64_t known_elements = 1;
    
    for (size_t i = 0; i < shape.size(); ++i) {
        int64_t dim = shape[i].expect_int();
        if (dim == -1) {
            TORCH_CHECK(infer_idx == -1, "reshape: only one dimension can be inferred");
            infer_idx = static_cast<int64_t>(i);
            new_shape.push_back(-1);  // 占位符
        } else {
            TORCH_CHECK(dim >= 0, "reshape: size must be non-negative");
            new_shape.push_back(dim);
            known_elements *= dim;
        }
    }
    
    if (infer_idx != -1) {
        TORCH_CHECK(known_elements > 0 && total_elements % known_elements == 0,
                    "reshape: cannot infer dimension");
        new_shape[infer_idx] = total_elements / known_elements;
    }
    
    // 验证元素数匹配
    int64_t new_total = 1;
    for (auto dim : new_shape) {
        new_total *= dim;
    }
    TORCH_CHECK(new_total == total_elements,
                "reshape: shape does not match number of elements");
    
    // 尝试作为 view 操作（如果 tensor 是 contiguous 的）
    if (self.is_contiguous()) {
        // 计算新的 strides
        std::vector<c10::SymInt> new_strides;
        new_strides.resize(new_shape.size());
        
        if (!new_shape.empty()) {
            new_strides[new_shape.size() - 1] = c10::SymInt(1);
            for (int64_t i = static_cast<int64_t>(new_shape.size()) - 2; i >= 0; --i) {
                new_strides[i] = c10::SymInt(new_strides[i + 1].expect_int() * new_shape[i + 1]);
            }
        }
        
        std::vector<c10::SymInt> sym_shape;
        for (auto dim : new_shape) {
            sym_shape.push_back(c10::SymInt(dim));
        }
        
        return cas_npu_as_strided(
            self,
            c10::SymIntArrayRef(sym_shape),
            c10::SymIntArrayRef(new_strides),
            c10::SymInt(self.storage_offset())
        );
    }
    
    // 如果不是 contiguous，需要先 clone 再 view
    auto contiguous_self = self.contiguous();
    
    std::vector<c10::SymInt> new_strides;
    new_strides.resize(new_shape.size());
    
    if (!new_shape.empty()) {
        new_strides[new_shape.size() - 1] = c10::SymInt(1);
        for (int64_t i = static_cast<int64_t>(new_shape.size()) - 2; i >= 0; --i) {
            new_strides[i] = c10::SymInt(new_strides[i + 1].expect_int() * new_shape[i + 1]);
        }
    }
    
    std::vector<c10::SymInt> sym_shape;
    for (auto dim : new_shape) {
        sym_shape.push_back(c10::SymInt(dim));
    }
    
    return cas_npu_as_strided(
        contiguous_self,
        c10::SymIntArrayRef(sym_shape),
        c10::SymIntArrayRef(new_strides),
        c10::SymInt(0)
    );
}

// ============ 数学操作实现 ============

// rsqrt实现 - 返回平方根的倒数 (1/sqrt(x))
at::Tensor cas_npu_rsqrt(const at::Tensor& self) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    // 创建输出tensor
    auto output = at::empty(self.sizes(), self.options());
    
    // 将设备tensor视为CPU tensor来执行操作
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    
    // 使用CPU实现
    at::rsqrt_out(out_as_cpu, self_as_cpu);
    
    return output;
}

// sqrt实现 - 返回平方根
at::Tensor cas_npu_sqrt(const at::Tensor& self) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    auto output = at::empty(self.sizes(), self.options());
    
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    
    at::sqrt_out(out_as_cpu, self_as_cpu);
    
    return output;
}

// mul.Tensor实现 - 逐元素乘法
at::Tensor cas_npu_mul_Tensor(const at::Tensor& self, const at::Tensor& other) {
    // 至少有一个 tensor 必须在 CAS-NPU 上
    bool self_on_device = self.device().is_privateuseone();
    bool other_on_device = other.device().is_privateuseone();
    TORCH_CHECK(self_on_device || other_on_device, 
                "Expected at least one tensor on CAS-NPU device");
    
    // 处理混合设备情况（如 tensor * scalar_tensor）
    // 如果 other 是 CPU 上的标量 tensor，将其值提取出来进行计算
    if (self_on_device && !other_on_device && other.numel() == 1) {
        // other 是 CPU 上的标量，转换为 float 并进行标量乘法
        float scalar_val = other.item<float>();
        auto output = at::empty(self.sizes(), self.options());
        
        float* out_data = output.data_ptr<float>();
        const float* self_data = self.data_ptr<float>();
        int64_t numel = self.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_data[i] = self_data[i] * scalar_val;
        }
        return output;
    }
    
    // 如果 self 是 CPU 上的标量，other 在设备上
    if (!self_on_device && other_on_device && self.numel() == 1) {
        float scalar_val = self.item<float>();
        auto output = at::empty(other.sizes(), other.options());
        
        float* out_data = output.data_ptr<float>();
        const float* other_data = other.data_ptr<float>();
        int64_t numel = other.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_data[i] = scalar_val * other_data[i];
        }
        return output;
    }
    
    // 两个 tensor 都在 CAS-NPU 上
    TORCH_CHECK(self_on_device && other_on_device,
                "Both tensors must be on CAS-NPU device for element-wise mul");
    
    // 处理广播
    auto output_size = at::infer_size(self.sizes(), other.sizes());
    auto output = at::empty(output_size, self.options());
    
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    at::Tensor other_as_cpu = at::from_blob(
        other.data_ptr(),
        other.sizes(),
        other.strides(),
        other.options().device(at::kCPU));
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    
    at::mul_out(out_as_cpu, self_as_cpu, other_as_cpu);
    
    return output;
}

// mul.Scalar实现 - 标量乘法
at::Tensor cas_npu_mul_Scalar(const at::Tensor& self, const at::Scalar& other) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    auto output = at::empty(self.sizes(), self.options());
    
    // 获取数据指针
    float* out_data = output.data_ptr<float>();
    const float* self_data = self.data_ptr<float>();
    float scalar_val = other.toFloat();
    
    // 直接执行标量乘法
    int64_t numel = self.numel();
    for (int64_t i = 0; i < numel; ++i) {
        out_data[i] = self_data[i] * scalar_val;
    }
    
    return output;
}

// pow.Tensor_Scalar实现 - 幂运算
at::Tensor cas_npu_pow_Tensor_Scalar(const at::Tensor& self, const at::Scalar& exponent) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    auto output = at::empty(self.sizes(), self.options());
    
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    
    at::pow_out(out_as_cpu, self_as_cpu, exponent);
    
    return output;
}

// mean.dim实现 - 沿维度求均值
at::Tensor cas_npu_mean_dim(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<c10::ScalarType> dtype) {
    
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    
    // 先在 CPU 上计算结果形状
    at::Tensor cpu_result = at::mean(self_as_cpu, dim, keepdim, dtype);
    
    // 在设备上创建输出tensor
    auto output = at::empty(cpu_result.sizes(), self.options());
    
    // 将结果拷贝到设备tensor
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    out_as_cpu.copy_(cpu_result);
    
    return output;
}

// silu实现 - SiLU激活函数 (x * sigmoid(x))
at::Tensor cas_npu_silu(const at::Tensor& self) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    auto output = at::empty(self.sizes(), self.options());
    
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    
    at::silu_out(out_as_cpu, self_as_cpu);
    
    return output;
}

// cos实现 - 余弦函数
at::Tensor cas_npu_cos(const at::Tensor& self) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    auto output = at::empty(self.sizes(), self.options());
    
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    
    at::cos_out(out_as_cpu, self_as_cpu);
    
    return output;
}

// sin实现 - 正弦函数
at::Tensor cas_npu_sin(const at::Tensor& self) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    auto output = at::empty(self.sizes(), self.options());
    
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    
    at::sin_out(out_as_cpu, self_as_cpu);
    
    return output;
}

// add.Scalar实现 - tensor与标量相加
at::Tensor cas_npu_add_Scalar(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    auto output = at::empty(self.sizes(), self.options());
    
    // 获取数据指针
    float* out_data = output.data_ptr<float>();
    const float* self_data = self.data_ptr<float>();
    float add_val = other.toFloat() * alpha.toFloat();
    
    // 直接执行标量加法
    int64_t numel = self.numel();
    for (int64_t i = 0; i < numel; ++i) {
        out_data[i] = self_data[i] + add_val;
    }
    
    return output;
}

// sub.Tensor实现 - tensor减法
at::Tensor cas_npu_sub_Tensor(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(other.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    auto output_size = at::infer_size(self.sizes(), other.sizes());
    auto output = at::empty(output_size, self.options());
    
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    at::Tensor other_as_cpu = at::from_blob(
        other.data_ptr(),
        other.sizes(),
        other.strides(),
        other.options().device(at::kCPU));
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    
    at::sub_out(out_as_cpu, self_as_cpu, other_as_cpu, alpha);
    
    return output;
}

// div.Tensor实现 - tensor除法
at::Tensor cas_npu_div_Tensor(const at::Tensor& self, const at::Tensor& other) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(other.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    auto output_size = at::infer_size(self.sizes(), other.sizes());
    auto output = at::empty(output_size, self.options());
    
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    at::Tensor other_as_cpu = at::from_blob(
        other.data_ptr(),
        other.sizes(),
        other.strides(),
        other.options().device(at::kCPU));
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    
    at::div_out(out_as_cpu, self_as_cpu, other_as_cpu);
    
    return output;
}

// neg实现 - 取负
at::Tensor cas_npu_neg(const at::Tensor& self) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    auto output = at::empty(self.sizes(), self.options());
    
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    at::Tensor out_as_cpu = at::from_blob(
        output.data_ptr(),
        output.sizes(),
        output.strides(),
        output.options().device(at::kCPU));
    
    at::neg_out(out_as_cpu, self_as_cpu);
    
    return output;
}

// mul.out实现 - tensor乘法输出到指定位置
at::Tensor& cas_npu_mul_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(other.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(out.device().is_privateuseone(), 
                "Expected output tensor on CAS-NPU device");
    
    at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));
    at::Tensor other_as_cpu = at::from_blob(
        other.data_ptr(),
        other.sizes(),
        other.strides(),
        other.options().device(at::kCPU));
    at::Tensor out_as_cpu = at::from_blob(
        out.data_ptr(),
        out.sizes(),
        out.strides(),
        out.options().device(at::kCPU));
    
    at::mul_out(out_as_cpu, self_as_cpu, other_as_cpu);
    
    return out;
}

// mul.Scalar_out实现 - 标量乘法输出到指定位置
at::Tensor& cas_npu_mul_Scalar_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(out.device().is_privateuseone(), 
                "Expected output tensor on CAS-NPU device");
    
    float* out_data = out.data_ptr<float>();
    const float* self_data = self.data_ptr<float>();
    float scalar_val = other.toFloat();
    
    int64_t numel = self.numel();
    for (int64_t i = 0; i < numel; ++i) {
        out_data[i] = self_data[i] * scalar_val;
    }
    
    return out;
}

// ============ 操作注册 ============

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // ========== 自定义实现的操作 ==========
    m.impl("add.Tensor", &cas_npu_add_Tensor);
    m.impl("mm", &cas_npu_mm);
    m.impl("bmm", &cas_npu_bmm);
    m.impl("cat", &cas_npu_cat);
    m.impl("cat.out", &cas_npu_cat_out);
    
    // 数学操作（原生实现，结果保持在设备上）
    m.impl("rsqrt", &cas_npu_rsqrt);
    m.impl("sqrt", &cas_npu_sqrt);
    m.impl("mul.Tensor", &cas_npu_mul_Tensor);
    m.impl("mul.Scalar", &cas_npu_mul_Scalar);
    m.impl("pow.Tensor_Scalar", &cas_npu_pow_Tensor_Scalar);
    m.impl("mean.dim", &cas_npu_mean_dim);
    m.impl("silu", &cas_npu_silu);
    m.impl("cos", &cas_npu_cos);
    m.impl("sin", &cas_npu_sin);
    m.impl("add.Scalar", &cas_npu_add_Scalar);
    m.impl("sub.Tensor", &cas_npu_sub_Tensor);
    m.impl("div.Tensor", &cas_npu_div_Tensor);
    m.impl("neg", &cas_npu_neg);
    m.impl("mul.out", &cas_npu_mul_out);
    m.impl("mul.Scalar_out", &cas_npu_mul_Scalar_out);
    
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
    
    // 视图操作（必须原生实现，不能使用CPU fallback）
    m.impl("unsqueeze", &cas_npu_unsqueeze);
    m.impl("squeeze", &cas_npu_squeeze);
    m.impl("squeeze.dim", &cas_npu_squeeze_dim);
    m.impl("expand", &cas_npu_expand);
    m.impl("slice.Tensor", &cas_npu_slice_Tensor);
    m.impl("select.int", &cas_npu_select_int);
    m.impl("transpose.int", &cas_npu_transpose_int);
    m.impl("permute", &cas_npu_permute);
    m.impl("reshape", &cas_npu_reshape);
    
    // ========== 使用CPU Fallback的操作 ==========
    
    // 基础数学操作
    // mul.Tensor, mul.Scalar, pow.Tensor_Scalar, sub.Tensor, div.Tensor, neg 已在上面原生实现
    
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
    // mean.dim 已在上面原生实现
    m.impl("var", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());  // LayerNorm可能需要
    m.impl("var.correction", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("std", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("max", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("max.dim", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("min", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("min.dim", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("argmax", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("argmin", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // Tensor操作
    m.impl("clone", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("contiguous", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    // 视图操作（expand, squeeze, unsqueeze, slice.Tensor, select.int）已在上面原生实现
    // 注意：copy_不能使用CPU fallback，会导致递归调用
    m.impl("index_select", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());  // Embedding需要
    m.impl("embedding", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());  // 直接注册embedding
    // transpose.int, permute, reshape 已在上面原生实现
    m.impl("flatten.using_ints", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("unflatten.int", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
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
    // silu 已在上面原生实现
    m.impl("silu_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("silu_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("gelu", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());  // BERT/GPT使用
    m.impl("gelu_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("gelu_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("leaky_relu", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("leaky_relu_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("leaky_relu_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 三角函数和其他数学函数
    // cos, sin, rsqrt, sqrt 已在上面原生实现
    m.impl("cos.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sin.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("rsqrt.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sqrt.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("exp", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("exp.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("reciprocal", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("reciprocal.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 矩阵运算
    // mm 和 bmm 已在上面实现，这里移除fallback注册
    // m.impl("mm", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("addmm", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    // m.impl("bmm", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // Softmax和Attention相关
    m.impl("softmax.int", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());  // Attention需要
    m.impl("_softmax", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_softmax_backward_data", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("scaled_dot_product_attention", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());  // Flash attention
    
    // Loss相关
    m.impl("log_softmax.int", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_log_softmax_backward_data", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("nll_loss_forward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("nll_loss_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mse_loss", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mse_loss_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("cross_entropy_loss", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 打印相关和检查操作
    m.impl("abs", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("abs.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isnan", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isinf", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isfinite", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("ne.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("eq.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("ne.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("eq.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("gt.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("gt.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("lt.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("lt.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("ge.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("ge.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("le.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("le.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // where 和 mask 操作
    m.impl("where.self", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("masked_fill.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("masked_fill.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    // 注意：to.dtype, to.device, _to_copy 不能使用 CPU fallback
    // 它们需要特殊处理或使用 PyTorch 默认实现（通过 empty + copy_）
    
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

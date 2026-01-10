// CAS-NPU Operations Implementation
// 注册操作到PrivateUse1和AutogradPrivateUse1 dispatch key
// 
// 内存模型：假设 CAS-NPU 是独立设备（类似 CUDA GPU），
// 设备内存和 CPU 内存不共享，需要显式进行数据拷贝。

#include "runtime/cas_npu_runtime.h"
#include "runtime/cas_npu_debug.h"

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
    // cpu_fallback 内部会调用 _copy_from 进行数据拷贝，传输会被统一捕获
    CAS_NPU_DEBUG_OP(debug::OpType::PURE_FALLBACK, op.schema().name().c_str(), "");
    at::native::cpu_fallback(op, stack);
}

// ============ 设备内存拷贝辅助函数 ============
// 注：大部分 Fallback 已改用 cpu_fallback，仅保留 add.Tensor 广播情况需要的辅助函数

// 将 CAS-NPU 设备上的 tensor 数据拷贝到 CPU tensor（仅用于 add.Tensor 广播）
// 此函数要求输入tensor是contiguous的
at::Tensor device_to_cpu(const at::Tensor& device_tensor) {
    TORCH_CHECK(device_tensor.device().is_privateuseone(),
                "Expected tensor on CAS-NPU device");
    TORCH_CHECK(device_tensor.is_contiguous(),
                "device_to_cpu requires contiguous tensor. Call .contiguous() first.");
    
    auto cpu_tensor = at::empty(
        device_tensor.sizes(),
        device_tensor.options().device(at::kCPU));
    
    size_t nbytes = device_tensor.numel() * device_tensor.element_size();
    if (nbytes > 0) {
        auto err = casNpuMemcpy(
            cpu_tensor.data_ptr(),
            device_tensor.data_ptr(),
            nbytes,
            CAS_NPU_MEMCPY_DEVICE_TO_HOST
        );
        TORCH_CHECK(err == CAS_NPU_SUCCESS,
                    "Device to CPU memcpy failed: ", casNpuGetErrorString(err));
    }
    
    return cpu_tensor;
}

// 将 CPU tensor 的数据拷贝到 CAS-NPU 设备上（仅用于 add.Tensor 广播）
// 注意：此函数假设输入tensor已经是contiguous的
at::Tensor cpu_to_device(const at::Tensor& cpu_tensor, c10::Device target_device) {
    TORCH_CHECK(cpu_tensor.is_cpu(), "Expected tensor on CPU");
    TORCH_CHECK(target_device.is_privateuseone(), "Expected target device to be CAS-NPU");
    
    // 对于非contiguous的CPU tensor，先调用CPU的contiguous()使其连续
    // 这不会导致递归，因为这里调用的是CPU实现
    auto cpu_tensor_contig = cpu_tensor.is_contiguous() ? cpu_tensor : cpu_tensor.contiguous();
    
    auto device_tensor = at::empty(
        cpu_tensor_contig.sizes(),
        cpu_tensor_contig.options().device(target_device));
    
    size_t nbytes = cpu_tensor_contig.numel() * cpu_tensor_contig.element_size();
    if (nbytes > 0) {
        auto err = casNpuMemcpy(
            device_tensor.data_ptr(),
            cpu_tensor_contig.data_ptr(),
            nbytes,
            CAS_NPU_MEMCPY_HOST_TO_DEVICE
        );
        TORCH_CHECK(err == CAS_NPU_SUCCESS,
                    "CPU to device memcpy failed: ", casNpuGetErrorString(err));
    }
    
    return device_tensor;
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
    
    // 确定目标设备
    c10::Device target_device = self_on_device ? self.device() : other.device();
    
    // 检查类型：如果任何一个 tensor 不是 float 类型，需要 fallback 到 CPU
    // 注意：CAS-NPU 目前只支持 float 类型，其他类型（bool, int等）需要 fallback
    bool need_fallback = (self.scalar_type() != at::kFloat || other.scalar_type() != at::kFloat);
    
    if (need_fallback) {
        CAS_NPU_DEBUG_OP(debug::OpType::CPU_FALLBACK, "add.Tensor", " (non-float type)");
        // 将设备上的 tensor 移到 CPU，CPU 上的 tensor 保持不动
        at::Tensor self_cpu = self_on_device ? 
            device_to_cpu(self.is_contiguous() ? self : self.contiguous()) : self;
        at::Tensor other_cpu = other_on_device ? 
            device_to_cpu(other.is_contiguous() ? other : other.contiguous()) : other;
        
        // 在 CPU 上执行加法
        at::Tensor result_cpu = at::add(self_cpu, other_cpu, alpha);
        
        // 对于非 float 类型，结果保持在 CPU 上
        // 如果调用者需要结果在设备上，它会自己处理（通过 .to(device)）
        // 这样可以避免类型不匹配的问题
        return result_cpu;
    }
    
    // 处理混合设备情况：将 CPU tensor copy 到设备（现在两个都是 float 类型）
    at::Tensor self_device = self_on_device ? self : cpu_to_device(self, target_device);
    at::Tensor other_device = other_on_device ? other : cpu_to_device(other, target_device);
    
    // 处理广播：先将数据 copy 到 CPU，使用 CPU 实现，再 copy 回设备
    auto output_size = at::infer_size(self_device.sizes(), other_device.sizes());
    
    if (self_device.sizes() != other_device.sizes()) {
        // 需要广播，使用 CPU 实现
        CAS_NPU_DEBUG_OP(debug::OpType::CPU_FALLBACK, "add.Tensor", " (broadcast)");
        // 确保输入tensor是contiguous的，因为device_to_cpu要求contiguous
        auto self_contig = self_device.is_contiguous() ? self_device : self_device.contiguous();
        auto other_contig = other_device.is_contiguous() ? other_device : other_device.contiguous();
        at::Tensor self_cpu = device_to_cpu(self_contig);
        at::Tensor other_cpu = device_to_cpu(other_contig);
        
        // 在 CPU 上执行带广播的加法
        at::Tensor result_cpu = at::add(self_cpu, other_cpu, alpha);
        
        // 将结果 copy 回设备
        return cpu_to_device(result_cpu, target_device);
    }
    
    // 不需要广播，直接在设备上执行
    CAS_NPU_DEBUG_OP(debug::OpType::NPU_NATIVE, "add.Tensor", " [%ld]", self_device.numel());
    
    // 重要：确保输入tensor是contiguous的！
    // 当tensor是transpose、slice等操作后的结果时，可能是非contiguous的
    // 直接使用data_ptr()会导致读取错误的数据（stride信息被忽略）
    auto self_contig = self_device.is_contiguous() ? self_device : self_device.contiguous();
    auto other_contig = other_device.is_contiguous() ? other_device : other_device.contiguous();
    
    auto output = at::empty(output_size, self_contig.options());
    
    float* out_data = output.data_ptr<float>();
    const float* self_data = self_contig.data_ptr<float>();
    const float* other_data = other_contig.data_ptr<float>();
    
    // 调用CAS-NPU runtime执行加法
    auto err = casNpuAddTensor(
        out_data,
        self_data,
        other_data,
        self_contig.numel(),
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
    
    TORCH_CHECK(input.dim() == 2, "input must be 2D tensor");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be 2D tensor");
    TORCH_CHECK(input.size(1) == mat2.size(0),
                "input size(1) must match mat2 size(0)");
    
    bool input_on_device = input.device().is_privateuseone();
    bool mat2_on_device = mat2.device().is_privateuseone();
    
    // 如果两个 tensor 都在 CPU 上，直接 fallback 到 CPU
    if (!input_on_device && !mat2_on_device) {
        CAS_NPU_DEBUG_OP(debug::OpType::CPU_FALLBACK, "mm", " (both on CPU)");
        return at::mm(input, mat2);
    }
    
    // 确定目标设备
    c10::Device target_device = input_on_device ? input.device() : mat2.device();
    
    // 检查类型：如果任何一个 tensor 不是 float 类型，需要 fallback 到 CPU
    bool need_fallback = (input.scalar_type() != at::kFloat || mat2.scalar_type() != at::kFloat);
    
    if (need_fallback) {
        CAS_NPU_DEBUG_OP(debug::OpType::CPU_FALLBACK, "mm", " (non-float type)");
        // 将设备上的 tensor 移到 CPU，CPU 上的 tensor 保持不动
        at::Tensor input_cpu = input_on_device ? 
            device_to_cpu(input.is_contiguous() ? input : input.contiguous()) : input;
        at::Tensor mat2_cpu = mat2_on_device ? 
            device_to_cpu(mat2.is_contiguous() ? mat2 : mat2.contiguous()) : mat2;
        
        // 在 CPU 上执行矩阵乘法
        at::Tensor result_cpu = at::mm(input_cpu, mat2_cpu);
        
        // 对于非 float 类型，结果保持在 CPU 上
        return result_cpu;
    }
    
    // 处理混合设备情况：将 CPU tensor copy 到设备（现在两个都是 float 类型）
    at::Tensor input_device = input_on_device ? input : cpu_to_device(input, target_device);
    at::Tensor mat2_device = mat2_on_device ? mat2 : cpu_to_device(mat2, target_device);
    
    // 重要：确保输入tensor是contiguous的！
    // 当tensor是transpose后的结果（如weight.t()）时，它是非contiguous的
    // 直接使用data_ptr()会导致读取错误的数据（stride信息被忽略）
    auto input_contig = input_device.is_contiguous() ? input_device : input_device.contiguous();
    auto mat2_contig = mat2_device.is_contiguous() ? mat2_device : mat2_device.contiguous();
    
    int64_t M = input_contig.size(0);
    int64_t K = input_contig.size(1);
    int64_t N = mat2_contig.size(1);
    
    CAS_NPU_DEBUG_OP(debug::OpType::NPU_NATIVE, "mm", " [%ldx%ld] @ [%ldx%ld]", M, K, K, N);
    
    auto output = at::empty({M, N}, input_contig.options());
    
    float* out_data = output.data_ptr<float>();
    const float* input_data = input_contig.data_ptr<float>();
    const float* mat2_data = mat2_contig.data_ptr<float>();
    
    auto err = casNpuMatMul(out_data, input_data, mat2_data, M, K, N);
    
    TORCH_CHECK(err == CAS_NPU_SUCCESS,
                "CAS-NPU mm operation failed: ", casNpuGetErrorString(err));
    
    return output;
}

// bmm实现 - 批量矩阵乘法
at::Tensor cas_npu_bmm(
    const at::Tensor& input,
    const at::Tensor& mat2) {
    
    TORCH_CHECK(input.dim() == 3, "input must be 3D tensor");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3D tensor");
    TORCH_CHECK(input.size(0) == mat2.size(0),
                "batch sizes must match");
    TORCH_CHECK(input.size(2) == mat2.size(1),
                "input size(2) must match mat2 size(1)");
    
    bool input_on_device = input.device().is_privateuseone();
    bool mat2_on_device = mat2.device().is_privateuseone();
    
    // 如果两个 tensor 都在 CPU 上，直接 fallback 到 CPU
    if (!input_on_device && !mat2_on_device) {
        CAS_NPU_DEBUG_OP(debug::OpType::CPU_FALLBACK, "bmm", " (both on CPU)");
        return at::bmm(input, mat2);
    }
    
    // 确定目标设备
    c10::Device target_device = input_on_device ? input.device() : mat2.device();
    
    // 检查类型：如果任何一个 tensor 不是 float 类型，需要 fallback 到 CPU
    bool need_fallback = (input.scalar_type() != at::kFloat || mat2.scalar_type() != at::kFloat);
    
    if (need_fallback) {
        CAS_NPU_DEBUG_OP(debug::OpType::CPU_FALLBACK, "bmm", " (non-float type)");
        // 将设备上的 tensor 移到 CPU，CPU 上的 tensor 保持不动
        at::Tensor input_cpu = input_on_device ? 
            device_to_cpu(input.is_contiguous() ? input : input.contiguous()) : input;
        at::Tensor mat2_cpu = mat2_on_device ? 
            device_to_cpu(mat2.is_contiguous() ? mat2 : mat2.contiguous()) : mat2;
        
        // 在 CPU 上执行批量矩阵乘法
        at::Tensor result_cpu = at::bmm(input_cpu, mat2_cpu);
        
        // 对于非 float 类型，结果保持在 CPU 上
        return result_cpu;
    }
    
    // 处理混合设备情况：将 CPU tensor copy 到设备（现在两个都是 float 类型）
    at::Tensor input_device = input_on_device ? input : cpu_to_device(input, target_device);
    at::Tensor mat2_device = mat2_on_device ? mat2 : cpu_to_device(mat2, target_device);
    
    // 重要：确保输入tensor是contiguous的！
    // 当tensor是transpose后的结果时，它是非contiguous的
    // 直接使用data_ptr()会导致读取错误的数据（stride信息被忽略）
    auto input_contig = input_device.is_contiguous() ? input_device : input_device.contiguous();
    auto mat2_contig = mat2_device.is_contiguous() ? mat2_device : mat2_device.contiguous();
    
    int64_t B = input_contig.size(0);
    int64_t M = input_contig.size(1);
    int64_t K = input_contig.size(2);
    int64_t N = mat2_contig.size(2);
    
    CAS_NPU_DEBUG_OP(debug::OpType::NPU_NATIVE, "bmm", " [%ldx%ldx%ld] @ [%ldx%ldx%ld]", B, M, K, B, K, N);
    
    auto output = at::empty({B, M, N}, input_contig.options());
    
    float* out_data = output.data_ptr<float>();
    const float* input_data = input_contig.data_ptr<float>();
    const float* mat2_data = mat2_contig.data_ptr<float>();
    
    auto err = casNpuBatchMatMul(out_data, input_data, mat2_data, B, M, K, N);
    
    TORCH_CHECK(err == CAS_NPU_SUCCESS,
                "CAS-NPU bmm operation failed: ", casNpuGetErrorString(err));
    
    return output;
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
// 这个函数会被 PyTorch 的 cpu_fallback 和 .to(device) 调用
at::Tensor cas_npu_copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
    
    TORCH_CHECK(self.defined(), "Source tensor is not defined");
    TORCH_CHECK(dst.defined(), "Destination tensor is not defined");
    
    // 【关键修复 1】检查数据类型是否相同
    // 如果数据类型不同，需要使用 PyTorch 的 copy_ 来处理类型转换
    // 直接 memcpy 不同类型的数据会导致问题
    if (self.scalar_type() != dst.scalar_type()) {
        CAS_NPU_DEBUG_OP(debug::OpType::CPU_FALLBACK, "_copy_from", " (dtype cast)");
        
        // 将 self 移到 CPU（如果不在的话）
        at::Tensor self_cpu;
        if (self.device().is_privateuseone()) {
            self_cpu = at::empty(self.sizes(), self.options().device(at::kCPU));
            size_t nbytes = self.numel() * self.element_size();
            if (nbytes > 0) {
                auto err = casNpuMemcpy(
                    self_cpu.data_ptr(),
                    self.data_ptr(),
                    nbytes,
                    CAS_NPU_MEMCPY_DEVICE_TO_HOST
                );
                TORCH_CHECK(err == CAS_NPU_SUCCESS, "Device to CPU copy failed");
            }
        } else {
            self_cpu = self;
        }
        
        // 在 CPU 上进行类型转换
        at::Tensor result_cpu = at::empty(dst.sizes(), dst.options().device(at::kCPU));
        result_cpu.copy_(self_cpu);  // PyTorch 会处理类型转换
        
        // 如果 dst 在 device 上，将结果拷贝回去
        if (dst.device().is_privateuseone()) {
            size_t nbytes = result_cpu.numel() * result_cpu.element_size();
            if (nbytes > 0) {
                auto err = casNpuMemcpy(
                    dst.data_ptr(),
                    result_cpu.data_ptr(),
                    nbytes,
                    CAS_NPU_MEMCPY_HOST_TO_DEVICE
                );
                TORCH_CHECK(err == CAS_NPU_SUCCESS, "Host to device copy failed");
            }
        } else {
            dst.copy_(result_cpu);
        }
        return dst;
    }
    
    // 对于非contiguous tensor，需要特殊处理
    if (!self.is_contiguous()) {
        CAS_NPU_DEBUG_OP(debug::OpType::CPU_FALLBACK, "_copy_from", " (source non-contiguous)");
        auto self_contig = self.contiguous();
        return cas_npu_copy_from(self_contig, dst, non_blocking);
    }
    
    if (!dst.is_contiguous()) {
        CAS_NPU_DEBUG_OP(debug::OpType::CPU_FALLBACK, "_copy_from", " (dst non-contiguous)");
        
        // 【关键修复 2】非 contiguous dst 的正确处理
        // 需要先将数据拷贝到 contiguous 的临时 tensor，
        // 然后通过 CPU 的 copy_ 来处理 stride
        
        if (self.is_cpu() && dst.is_cpu()) {
            // 两者都在 CPU，直接使用 PyTorch 的 copy_
                dst.copy_(self);
                return dst;
            }
        
        // 将 self 移到 CPU
        at::Tensor self_cpu;
        if (self.device().is_privateuseone()) {
            self_cpu = at::empty(self.sizes(), self.options().device(at::kCPU));
        size_t nbytes = self.numel() * self.element_size();
        if (nbytes > 0) {
            auto err = casNpuMemcpy(
                self_cpu.data_ptr(),
                self.data_ptr(),
                nbytes,
                CAS_NPU_MEMCPY_DEVICE_TO_HOST
            );
                TORCH_CHECK(err == CAS_NPU_SUCCESS, "Device to CPU copy failed");
            }
        } else {
            self_cpu = self;
        }
        
        if (dst.is_cpu()) {
            // dst 在 CPU，使用 copy_ 处理 stride
            dst.copy_(self_cpu);
            return dst;
        }
        
        // dst 在 device 上且非 contiguous
        // 【关键修复】这种情况需要特殊处理
        // 先将 dst 拷贝到 CPU，使用 copy_ 更新数据，再拷贝回去
        at::Tensor dst_cpu = at::empty(dst.sizes(), dst.options().device(at::kCPU));
        // 先拷贝原有数据（保留可能需要的部分）
        size_t dst_nbytes = dst.numel() * dst.element_size();
        if (dst_nbytes > 0) {
            auto err = casNpuMemcpy(
                dst_cpu.data_ptr(),
                dst.data_ptr(),
                dst_nbytes,
                CAS_NPU_MEMCPY_DEVICE_TO_HOST
            );
            TORCH_CHECK(err == CAS_NPU_SUCCESS, "Device to CPU copy failed");
        }
        // 使用 CPU 的 copy_ 更新数据（会正确处理 stride）
        dst_cpu.copy_(self_cpu);
        // 将更新后的数据拷贝回 device
        if (dst_nbytes > 0) {
            auto err = casNpuMemcpy(
                dst.data_ptr(),
                dst_cpu.data_ptr(),
                dst_nbytes,
                CAS_NPU_MEMCPY_HOST_TO_DEVICE
            );
            TORCH_CHECK(err == CAS_NPU_SUCCESS, "Host to device copy failed");
    }
        return dst;
    }
    
    // self 和 dst 都是 contiguous，可以直接 memcpy
    size_t nbytes = self.numel() * self.element_size();
    
    // 同设备拷贝: Device -> Device
    if (self.device() == dst.device()) {
        if (nbytes > 0) {
            CAS_NPU_DEBUG_OP(debug::OpType::DATA_COPY, "_copy_from", " D→D %.2f KB", nbytes / 1024.0);
            if (debug::debug_level() >= 2) {
                auto& stats = debug::get_stats();
                stats.d2d_transfers++;
                stats.d2d_bytes += nbytes;
            }
            auto err = casNpuMemcpy(
                dst.data_ptr(),
                self.data_ptr(),
                nbytes,
                CAS_NPU_MEMCPY_DEVICE_TO_DEVICE
            );
            TORCH_CHECK(err == CAS_NPU_SUCCESS,
                        "Device to device copy failed: ", casNpuGetErrorString(err));
        }
    }
    // CPU -> Device
    else if (self.is_cpu()) {
        if (nbytes > 0) {
            CAS_NPU_DEBUG_OP(debug::OpType::DATA_COPY, "_copy_from", " H→D %.2f KB", nbytes / 1024.0);
            if (debug::debug_level() >= 2) {
                auto& stats = debug::get_stats();
                stats.h2d_transfers++;
                stats.h2d_bytes += nbytes;
            }
            auto err = casNpuMemcpy(
                dst.data_ptr(),
                self.data_ptr(),
                nbytes,
                CAS_NPU_MEMCPY_HOST_TO_DEVICE
            );
            TORCH_CHECK(err == CAS_NPU_SUCCESS,
                        "Host to device copy failed: ", casNpuGetErrorString(err));
        }
    }
    // Device -> CPU
    else {
        if (nbytes > 0) {
            CAS_NPU_DEBUG_OP(debug::OpType::DATA_COPY, "_copy_from", " D→H %.2f KB", nbytes / 1024.0);
            if (debug::debug_level() >= 2) {
                auto& stats = debug::get_stats();
                stats.d2h_transfers++;
                stats.d2h_bytes += nbytes;
            }
            auto err = casNpuMemcpy(
                dst.data_ptr(),
                self.data_ptr(),
                nbytes,
                CAS_NPU_MEMCPY_DEVICE_TO_HOST
            );
            TORCH_CHECK(err == CAS_NPU_SUCCESS,
                        "Device to host copy failed: ", casNpuGetErrorString(err));
        }
    }
    
    return dst;
}

// _copy_from_and_resize实现
at::Tensor cas_npu_copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
    
    at::native::resize_(dst, self.sizes(), std::nullopt);
    return cas_npu_copy_from(self, dst, false);
}

// view实现
at::Tensor cas_npu_view(const at::Tensor& self, c10::SymIntArrayRef size) {
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "view", "");
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

// _local_scalar_dense实现 - 读取设备上的单个标量值
at::Scalar cas_npu_local_scalar_dense(const at::Tensor& self) {
    TORCH_CHECK(self.numel() == 1, "Expected single element tensor");
    
    // 将单个元素从设备 copy 到 CPU
    // 确保tensor是contiguous的，因为device_to_cpu要求contiguous
    auto self_contig = self.is_contiguous() ? self : self.contiguous();
    at::Tensor cpu_tensor = device_to_cpu(self_contig);
    return at::native::_local_scalar_dense_cpu(cpu_tensor);
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
    
    return at::native::as_strided_tensorimpl(
        self,
        C10_AS_INTARRAYREF_SLOW(size),
        C10_AS_INTARRAYREF_SLOW(stride),
        storage_offset);
}

// detach实现
at::Tensor cas_npu_detach(const at::Tensor& self) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "detach", "");
    
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
    
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "t", "");
    
    int64_t M = self.size(0);
    int64_t N = self.size(1);
    auto strides = self.strides();
    
    std::vector<c10::SymInt> new_size = {c10::SymInt(N), c10::SymInt(M)};
    std::vector<c10::SymInt> new_stride = {c10::SymInt(strides[1]), c10::SymInt(strides[0])};
    
    return cas_npu_as_strided(
        self,
        c10::SymIntArrayRef(new_size),
        c10::SymIntArrayRef(new_stride),
        c10::SymInt(self.storage_offset())
    );
}

// ============ View 操作实现 ============
// 这些操作不涉及数据拷贝，只修改 tensor 的 metadata

// unsqueeze实现
at::Tensor cas_npu_unsqueeze(const at::Tensor& self, int64_t dim) {
    TORCH_CHECK(self.device().is_privateuseone(), 
                "Expected tensor on CAS-NPU device");
    
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "unsqueeze", " dim=%ld", dim);
    
    int64_t ndim = self.dim();
    if (dim < 0) dim = dim + ndim + 1;
    TORCH_CHECK(dim >= 0 && dim <= ndim, "unsqueeze: dimension out of range");
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    new_sizes.reserve(ndim + 1);
    new_strides.reserve(ndim + 1);
    
    for (int64_t i = 0; i < ndim + 1; ++i) {
        if (i == dim) {
            new_sizes.push_back(c10::SymInt(1));
            new_strides.push_back(i < ndim ? c10::SymInt(old_strides[i]) : c10::SymInt(1));
        } else {
            int64_t old_idx = (i < dim) ? i : i - 1;
            new_sizes.push_back(c10::SymInt(old_sizes[old_idx]));
            new_strides.push_back(c10::SymInt(old_strides[old_idx]));
        }
    }
    
    return cas_npu_as_strided(self, new_sizes, new_strides, c10::SymInt(self.storage_offset()));
}

// squeeze.dim实现
at::Tensor cas_npu_squeeze_dim(const at::Tensor& self, int64_t dim) {
    TORCH_CHECK(self.device().is_privateuseone(), "Expected tensor on CAS-NPU device");
    
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "squeeze.dim", " dim=%ld", dim);
    
    int64_t ndim = self.dim();
    if (dim < 0) dim = dim + ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "squeeze: dimension out of range");
    
    if (self.size(dim) != 1) return self;
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            new_sizes.push_back(c10::SymInt(old_sizes[i]));
            new_strides.push_back(c10::SymInt(old_strides[i]));
        }
    }
    
    return cas_npu_as_strided(self, new_sizes, new_strides, c10::SymInt(self.storage_offset()));
}

// squeeze实现
at::Tensor cas_npu_squeeze(const at::Tensor& self) {
    TORCH_CHECK(self.device().is_privateuseone(), "Expected tensor on CAS-NPU device");
    
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "squeeze", "");
    
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
    
    if (new_sizes.size() == static_cast<size_t>(ndim)) return self;
    
    return cas_npu_as_strided(self, new_sizes, new_strides, c10::SymInt(self.storage_offset()));
}

// expand实现
at::Tensor cas_npu_expand(const at::Tensor& self, c10::SymIntArrayRef size, bool implicit) {
    TORCH_CHECK(self.device().is_privateuseone(), "Expected tensor on CAS-NPU device");
    
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "expand", "");
    
    int64_t ndim = size.size();
    int64_t self_ndim = self.dim();
    TORCH_CHECK(ndim >= self_ndim, "expand: insufficient dimensions");
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    int64_t offset = ndim - self_ndim;
    
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t size_val = size[i].expect_int();
        
        if (i < offset) {
            TORCH_CHECK(size_val >= 0, "expand: size must be non-negative");
            new_sizes.push_back(size[i]);
            new_strides.push_back(c10::SymInt(0));
        } else {
            int64_t old_idx = i - offset;
            int64_t old_size = old_sizes[old_idx];
            int64_t old_stride = old_strides[old_idx];
            
            if (size_val == -1) {
                new_sizes.push_back(c10::SymInt(old_size));
                new_strides.push_back(c10::SymInt(old_stride));
            } else if (old_size == 1) {
                new_sizes.push_back(size[i]);
                new_strides.push_back(c10::SymInt(0));
            } else if (old_size == size_val) {
                new_sizes.push_back(c10::SymInt(old_size));
                new_strides.push_back(c10::SymInt(old_stride));
            } else {
                TORCH_CHECK(false, "expand: size mismatch at dimension ", i);
            }
        }
    }
    
    return cas_npu_as_strided(self, new_sizes, new_strides, c10::SymInt(self.storage_offset()));
}

// slice.Tensor实现
at::Tensor cas_npu_slice_Tensor(
    const at::Tensor& self, int64_t dim,
    c10::optional<c10::SymInt> start_opt, c10::optional<c10::SymInt> end_opt,
    c10::SymInt step) {
    
    TORCH_CHECK(self.device().is_privateuseone(), "Expected tensor on CAS-NPU device");
    
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "slice.Tensor", " dim=%ld", dim);
    
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "slice: dimension out of range");
    
    int64_t dim_size = self.size(dim);
    int64_t step_val = step.expect_int();
    TORCH_CHECK(step_val > 0, "slice step must be positive");
    
    int64_t start_val = start_opt.has_value() ? start_opt.value().expect_int() : 0;
    if (start_val < 0) start_val += dim_size;
    start_val = std::max(int64_t(0), std::min(start_val, dim_size));
    
    int64_t end_val = end_opt.has_value() ? end_opt.value().expect_int() : dim_size;
    if (end_val < 0) end_val += dim_size;
    end_val = std::max(int64_t(0), std::min(end_val, dim_size));
    
    int64_t slice_size = (end_val > start_val) ? (end_val - start_val + step_val - 1) / step_val : 0;
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            new_sizes.push_back(c10::SymInt(slice_size));
            new_strides.push_back(c10::SymInt(old_strides[i] * step_val));
        } else {
            new_sizes.push_back(c10::SymInt(old_sizes[i]));
            new_strides.push_back(c10::SymInt(old_strides[i]));
        }
    }
    
    int64_t new_offset = self.storage_offset() + start_val * old_strides[dim];
    return cas_npu_as_strided(self, new_sizes, new_strides, c10::SymInt(new_offset));
}

// select.int实现
at::Tensor cas_npu_select_int(const at::Tensor& self, int64_t dim, int64_t index) {
    TORCH_CHECK(self.device().is_privateuseone(), "Expected tensor on CAS-NPU device");
    
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "select.int", " dim=%ld idx=%ld", dim, index);
    
    int64_t ndim = self.dim();
    TORCH_CHECK(ndim > 0, "select: cannot select on 0-d tensor");
    
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "select: dimension out of range");
    
    int64_t dim_size = self.size(dim);
    if (index < 0) index += dim_size;
    TORCH_CHECK(index >= 0 && index < dim_size, "select: index out of range");
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            new_sizes.push_back(c10::SymInt(old_sizes[i]));
            new_strides.push_back(c10::SymInt(old_strides[i]));
        }
    }
    
    int64_t new_offset = self.storage_offset() + index * old_strides[dim];
    return cas_npu_as_strided(self, new_sizes, new_strides, c10::SymInt(new_offset));
}

// transpose.int实现
at::Tensor cas_npu_transpose_int(const at::Tensor& self, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(self.device().is_privateuseone(), "Expected tensor on CAS-NPU device");
    
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "transpose.int", " dim0=%ld dim1=%ld", dim0, dim1);
    
    int64_t ndim = self.dim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    TORCH_CHECK(dim0 >= 0 && dim0 < ndim && dim1 >= 0 && dim1 < ndim, "transpose: dimension out of range");
    
    if (dim0 == dim1) return self;
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t src = (i == dim0) ? dim1 : ((i == dim1) ? dim0 : i);
        new_sizes.push_back(c10::SymInt(old_sizes[src]));
        new_strides.push_back(c10::SymInt(old_strides[src]));
    }
    
    return cas_npu_as_strided(self, new_sizes, new_strides, c10::SymInt(self.storage_offset()));
}

// permute实现
at::Tensor cas_npu_permute(const at::Tensor& self, c10::IntArrayRef dims) {
    TORCH_CHECK(self.device().is_privateuseone(), "Expected tensor on CAS-NPU device");
    
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "permute", "");
    
    int64_t ndim = self.dim();
    TORCH_CHECK(static_cast<int64_t>(dims.size()) == ndim, "permute: dims size mismatch");
    
    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    
    std::vector<bool> seen(ndim, false);
    std::vector<c10::SymInt> new_sizes;
    std::vector<c10::SymInt> new_strides;
    
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t d = dims[i];
        if (d < 0) d += ndim;
        TORCH_CHECK(d >= 0 && d < ndim && !seen[d], "permute: invalid dimension");
        seen[d] = true;
        new_sizes.push_back(c10::SymInt(old_sizes[d]));
        new_strides.push_back(c10::SymInt(old_strides[d]));
    }
    
    return cas_npu_as_strided(self, new_sizes, new_strides, c10::SymInt(self.storage_offset()));
}

// reshape实现
at::Tensor cas_npu_reshape(const at::Tensor& self, c10::SymIntArrayRef shape) {
    TORCH_CHECK(self.device().is_privateuseone(), "Expected tensor on CAS-NPU device");
    
    CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "reshape", "");
    
    int64_t total_elements = self.numel();
    std::vector<int64_t> new_shape;
    int64_t infer_idx = -1;
    int64_t known_elements = 1;
    
    for (size_t i = 0; i < shape.size(); ++i) {
        int64_t dim = shape[i].expect_int();
        if (dim == -1) {
            TORCH_CHECK(infer_idx == -1, "reshape: only one dimension can be inferred");
            infer_idx = static_cast<int64_t>(i);
            new_shape.push_back(-1);
        } else {
            TORCH_CHECK(dim >= 0, "reshape: size must be non-negative");
            new_shape.push_back(dim);
            known_elements *= dim;
        }
    }
    
    if (infer_idx != -1) {
        TORCH_CHECK(known_elements > 0 && total_elements % known_elements == 0, "reshape: cannot infer dimension");
        new_shape[infer_idx] = total_elements / known_elements;
    }
    
    int64_t new_total = 1;
    for (auto dim : new_shape) new_total *= dim;
    TORCH_CHECK(new_total == total_elements, "reshape: shape mismatch");
    
    if (self.is_contiguous()) {
        std::vector<c10::SymInt> new_strides(new_shape.size());
        if (!new_shape.empty()) {
            new_strides[new_shape.size() - 1] = c10::SymInt(1);
            for (int64_t i = static_cast<int64_t>(new_shape.size()) - 2; i >= 0; --i) {
                new_strides[i] = c10::SymInt(new_strides[i + 1].expect_int() * new_shape[i + 1]);
            }
        }
        
        std::vector<c10::SymInt> sym_shape;
        for (auto dim : new_shape) sym_shape.push_back(c10::SymInt(dim));
        
        return cas_npu_as_strided(self, sym_shape, new_strides, c10::SymInt(self.storage_offset()));
    }
    
    // 非 contiguous 需要先 copy 到 contiguous 布局
    auto contiguous_self = self.contiguous();
    return cas_npu_reshape(contiguous_self, shape);
}

// contiguous实现 - 返回内存连续的tensor（保留优化：已经contiguous时直接返回）
at::Tensor cas_npu_contiguous(
    const at::Tensor& self,
    at::MemoryFormat memory_format) {
    
    TORCH_CHECK(self.device().is_privateuseone(), "Expected tensor on CAS-NPU device");
    
    // 如果已经是 contiguous 的，直接返回
    if (self.is_contiguous(memory_format)) {
        CAS_NPU_DEBUG_OP(debug::OpType::VIEW_OP, "contiguous", " (already contiguous)");
        return self;
    }
    
    // 需要创建一个新的 contiguous tensor 并拷贝数据
    // 策略：通过CPU中转，避免D2D非contiguous拷贝的复杂性
    // 1. 复制整个storage到CPU
    // 2. 在CPU上创建相同strides的view
    // 3. 调用CPU的contiguous
    // 4. 复制回device
    CAS_NPU_DEBUG_OP(debug::OpType::CPU_FALLBACK, "contiguous", " (needs copy via CPU)");
    
    // 获取storage信息
    size_t storage_nbytes = self.storage().nbytes();
    size_t storage_offset = self.storage_offset();
    
    // 创建CPU storage并复制数据
    auto cpu_storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        storage_nbytes,
        at::getCPUAllocator(),
        true);
    
    if (storage_nbytes > 0) {
        auto err = casNpuMemcpy(
            cpu_storage.mutable_data(),
            self.storage().data(),
            storage_nbytes,
            CAS_NPU_MEMCPY_DEVICE_TO_HOST
        );
        TORCH_CHECK(err == CAS_NPU_SUCCESS,
                    "Device to CPU storage copy failed: ", casNpuGetErrorString(err));
    }
    
    // 在CPU上创建相同metadata的tensor view
    auto cpu_tensor = at::empty({0}, self.options().device(at::kCPU));
    cpu_tensor.set_(cpu_storage, storage_offset, self.sizes(), self.strides());
    
    // 调用CPU的contiguous（不会递归，因为在CPU上）
    auto contiguous_cpu = cpu_tensor.contiguous(memory_format);
    
    // 复制回device
    return cpu_to_device(contiguous_cpu, self.device());
}

// ============ masked_fill_ ============
// 注意：不提供自定义实现，直接使用 cpu_fallback
// 之前的自定义实现反而引入了问题

// ============ 操作注册 ============

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // ========== NPU 原生实现的操作 ==========
    m.impl("add.Tensor", &cas_npu_add_Tensor);
    m.impl("mm", &cas_npu_mm);
    m.impl("bmm", &cas_npu_bmm);
    
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
    
    // 视图操作（不涉及数据拷贝）
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
    
    // 数学操作 (通过 cpu_fallback 自动处理)
    m.impl("rsqrt", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sqrt", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mul.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mul.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("pow.Tensor_Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mean.dim", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("silu", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("cos", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sin", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("add.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sub.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("div.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("div.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("div.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("div.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("neg", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mul.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mul.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("cat", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("cat.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // Tensor工厂和操作
    m.impl("ones_like", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("zeros_like", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("full_like", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("fill_.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("zero_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("arange.start_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("arange.start_step", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("arange", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 规约操作
    m.impl("sum", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sum.dim_IntList", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("cumsum", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("cumsum.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("cumsum.dimname", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("cumsum.dimname_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mean", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("var", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("var.correction", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("std", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("max", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("max.dim", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("min", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("min.dim", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("argmax", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("argmin", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("topk", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("topk.values", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sort", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sort.values", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sort.values_stable", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sort.dimname", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sort.dimname_values", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sort.dimname_values_stable", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // Tensor操作
    m.impl("clone", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("contiguous", &cas_npu_contiguous);
    m.impl("index_select", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("embedding", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("gather", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("gather.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("gather.dimname", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("gather.dimname_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 索引操作 (CPU Fallback)
    m.impl("index.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("index.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("index_put_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("index_put", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("index_put.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("scatter", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("scatter.src", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("scatter.src_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("scatter.value", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("scatter.value_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("scatter.dimname", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("scatter.dimname_src", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("scatter.dimname_value", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
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
    m.impl("silu_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("silu_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("gelu", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("gelu_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("gelu_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("leaky_relu", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("leaky_relu_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("leaky_relu_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // Dropout 相关操作 (CPU Fallback)
    m.impl("native_dropout", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("native_dropout_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("dropout", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("dropout_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());

    // 三角函数和其他数学函数
    m.impl("cos.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sin.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("rsqrt.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sqrt.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("exp", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("exp.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("reciprocal", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("reciprocal.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 线性插值操作 (优化器中使用，如 AdamW)
    m.impl("lerp.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("lerp.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("lerp.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("lerp.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // Foreach 操作符 (优化器批量处理张量列表时使用)
    m.impl("_foreach_lerp_.ScalarList", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_foreach_lerp_.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_foreach_lerp.ScalarList", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_foreach_lerp.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_foreach_div_.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_foreach_div_.ScalarList", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_foreach_div.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_foreach_div.ScalarList", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 矩阵运算
    m.impl("addmm", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // Softmax和Attention相关
    m.impl("softmax.int", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_softmax", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_softmax_backward_data", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("scaled_dot_product_attention", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // Loss相关
    m.impl("log_softmax.int", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("_log_softmax_backward_data", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("nll_loss_forward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("nll_loss_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mse_loss", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mse_loss_backward", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("cross_entropy_loss", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());

    m.impl("normal_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("multinomial", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("multinomial.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 打印相关和检查操作
    m.impl("abs", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("abs.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isnan", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isinf", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isfinite", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isin.Tensor_Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isin.Tensor_Tensor_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isin.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("isin.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("ne.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("all", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("all.all_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("any", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("any.any_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 位运算操作 (CPU Fallback)
    m.impl("bitwise_and.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("bitwise_and.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("bitwise_and.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("bitwise_and.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("bitwise_or.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("bitwise_or.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("bitwise_xor.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("bitwise_xor.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("bitwise_not", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("bitwise_not.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
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
    m.impl("masked_select", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("masked_select.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("masked_fill", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("masked_fill.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("masked_fill.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    // inplace 版本也使用 cpu_fallback
    // 注意：虽然可能会有内存问题，但自定义实现似乎更容易出问题
    m.impl("masked_fill_.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("masked_fill_.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // SGD优化器相关
    m.impl("add.out", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("add_.Tensor", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("mul_.Scalar", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("addcmul_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("addcdiv_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    m.impl("sqrt_", torch::CppFunction::makeFromBoxedFunction<&cas_npu_fallback>());
    
    // 注意：PyTorch 的 PrivateUse1 后端不支持自动全局 fallback
    // 必须手动注册每个操作符。如果遇到新的未实现操作符，需要在这里添加
    // 建议：遇到新错误时，添加相应的 fallback 注册
}

// ============ AutogradPrivateUse1 注册 ============

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}

} // anonymous namespace

// CAS-NPU Allocator Implementation
#include "cas_npu_allocator.h"
#include "runtime/cas_npu_runtime.h"

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

namespace cas_npu {

namespace {

// 全局分配器实例
CasNpuAllocator g_cas_npu_allocator;

// 内存删除器函数
void deleteCasNpuMemory(void* ptr) {
    auto err = casNpuFree(ptr);
    if (err != CAS_NPU_SUCCESS) {
        TORCH_WARN("Failed to free CAS-NPU memory: ", casNpuGetErrorString(err));
    }
}

} // anonymous namespace

c10::DataPtr CasNpuAllocator::allocate(size_t nbytes) {
    // 获取当前设备
    int current_device_index = 0;
    auto err = casNpuGetDevice(&current_device_index);
    TORCH_CHECK(err == CAS_NPU_SUCCESS, 
                "Failed to get current CAS-NPU device: ", 
                casNpuGetErrorString(err));
    
    auto device = c10::Device(c10::DeviceType::PrivateUse1, current_device_index);
    
    void* data = nullptr;
    if (nbytes > 0) {
        err = casNpuMalloc(&data, nbytes);
        TORCH_CHECK(err == CAS_NPU_SUCCESS,
                    "Failed to allocate ", nbytes, " bytes on CAS-NPU device: ",
                    casNpuGetErrorString(err));
    }
    
    return {data, data, &deleteCasNpuMemory, device};
}

c10::DeleterFnPtr CasNpuAllocator::raw_deleter() const {
    return &deleteCasNpuMemory;
}

void CasNpuAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
    auto err = casNpuMemcpy(dest, src, count);
    TORCH_CHECK(err == CAS_NPU_SUCCESS,
                "Failed to copy ", count, " bytes on CAS-NPU device: ",
                casNpuGetErrorString(err));
}

CasNpuAllocator* getCasNpuAllocator() {
    return &g_cas_npu_allocator;
}

// 注册分配器到PrivateUse1
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_cas_npu_allocator);

} // namespace cas_npu


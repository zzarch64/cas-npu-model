// ECHO-NPU Allocator Implementation
#include "echo_npu_allocator.h"
#include "runtime/echo_npu_runtime.h"

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

namespace echo_npu {

namespace {

// 全局分配器实例
EchoNpuAllocator g_echo_npu_allocator;

// 内存删除器函数
void deleteEchoNpuMemory(void* ptr) {
    auto err = echoNpuFree(ptr);
    if (err != ECHO_NPU_SUCCESS) {
        TORCH_WARN("Failed to free ECHO-NPU memory: ", echoNpuGetErrorString(err));
    }
}

} // anonymous namespace

c10::DataPtr EchoNpuAllocator::allocate(size_t nbytes) {
    // 获取当前设备
    int current_device_index = 0;
    auto err = echoNpuGetDevice(&current_device_index);
    TORCH_CHECK(err == ECHO_NPU_SUCCESS, 
                "Failed to get current ECHO-NPU device: ", 
                echoNpuGetErrorString(err));
    
    auto device = c10::Device(c10::DeviceType::PrivateUse1, current_device_index);
    
    void* data = nullptr;
    if (nbytes > 0) {
        err = echoNpuMalloc(&data, nbytes);
        TORCH_CHECK(err == ECHO_NPU_SUCCESS,
                    "Failed to allocate ", nbytes, " bytes on ECHO-NPU device: ",
                    echoNpuGetErrorString(err));
    }
    
    return {data, data, &deleteEchoNpuMemory, device};
}

c10::DeleterFnPtr EchoNpuAllocator::raw_deleter() const {
    return &deleteEchoNpuMemory;
}

void EchoNpuAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
    // 这里假设是设备到设备的拷贝（分配器内部使用）
    auto err = echoNpuMemcpy(dest, src, count, ECHO_NPU_MEMCPY_DEVICE_TO_DEVICE);
    TORCH_CHECK(err == ECHO_NPU_SUCCESS,
                "Failed to copy ", count, " bytes on ECHO-NPU device: ",
                echoNpuGetErrorString(err));
}

EchoNpuAllocator* getEchoNpuAllocator() {
    return &g_echo_npu_allocator;
}

// 注册分配器到PrivateUse1
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_echo_npu_allocator);

} // namespace echo_npu


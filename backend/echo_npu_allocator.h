// ECHO-NPU Allocator Header
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

namespace echo_npu {

// ECHO-NPU设备内存分配器
class EchoNpuAllocator : public c10::Allocator {
public:
    EchoNpuAllocator() = default;
    ~EchoNpuAllocator() override = default;
    
    // 分配设备内存
    c10::DataPtr allocate(size_t nbytes) override;
    
    // 获取删除器函数指针
    c10::DeleterFnPtr raw_deleter() const override;
    
    // 复制数据
    void copy_data(void* dest, const void* src, std::size_t count) const override;
};

// 获取全局分配器实例
EchoNpuAllocator* getEchoNpuAllocator();

// 注册分配器
void registerEchoNpuAllocator();

} // namespace echo_npu


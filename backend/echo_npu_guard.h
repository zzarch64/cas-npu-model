// ECHO-NPU DeviceGuard Header
#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace echo_npu {

// ECHO-NPU DeviceGuard实现
// 负责设备切换、流管理等
struct EchoNpuGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;
    
    EchoNpuGuardImpl() = default;
    
    explicit EchoNpuGuardImpl(c10::DeviceType t) {
        TORCH_CHECK(t == static_type, 
                    "EchoNpuGuardImpl initialized with non-PrivateUse1 DeviceType: ", t);
    }
    
    // 返回设备类型
    c10::DeviceType type() const override;
    
    // 交换当前设备，返回之前的设备
    c10::Device exchangeDevice(c10::Device d) const override;
    
    // 获取当前设备
    c10::Device getDevice() const override;
    
    // 设置当前设备
    void setDevice(c10::Device d) const override;
    
    // 无检查设置设备（用于析构函数等）
    void uncheckedSetDevice(c10::Device d) const noexcept override;
    
    // 获取设备数量
    c10::DeviceIndex deviceCount() const noexcept override;
    
    // 获取指定设备的当前流
    c10::Stream getStream(c10::Device d) const noexcept override;
    
    // 获取默认流
    c10::Stream getDefaultStream(c10::Device d) const override;
    
    // 创建新流
    c10::Stream getNewStream(c10::Device d, int priority = 0) const override;
    
    // 从全局池获取流
    c10::Stream getStreamFromGlobalPool(c10::Device d, bool isHighPriority = false) const override;
    
    // 交换流
    c10::Stream exchangeStream(c10::Stream s) const noexcept override;
    
    // 查询流是否完成
    bool queryStream(const c10::Stream& stream) const override;
    
    // 同步流
    void synchronizeStream(const c10::Stream& stream) const override;
    
    // 同步设备
    void synchronizeDevice(const c10::DeviceIndex device_index) const override;
    
    // 事件相关
    void destroyEvent(void* event, const c10::DeviceIndex device_index) const noexcept override;
    
    void record(void** event, const c10::Stream& stream, 
                const c10::DeviceIndex device_index,
                const c10::EventFlag flag) const override;
    
    void block(void* event, const c10::Stream& stream) const override;
    
    bool queryEvent(void* event) const override;
    
    void synchronizeEvent(void* event) const override;
    
    double elapsedTime(void* event1, void* event2, 
                       const c10::DeviceIndex device_index) const override;
};

} // namespace echo_npu


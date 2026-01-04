// CAS-NPU Hooks Header
#pragma once

#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace cas_npu {

// CAS-NPU Hooks实现
// 提供设备相关的钩子函数
struct CasNpuHooksInterface : public at::PrivateUse1HooksInterface {
    CasNpuHooksInterface() = default;
    ~CasNpuHooksInterface() override = default;
    
    // 初始化设备
    void init() const override {}
    
    // 检查设备是否可用
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override;
    
    // 获取默认生成器（随机数）
    const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) const override;
};

// 注册Hooks
void registerCasNpuHooks();

} // namespace cas_npu


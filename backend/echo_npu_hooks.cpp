// ECHO-NPU Hooks Implementation
#include "echo_npu_hooks.h"
#include "runtime/echo_npu_runtime.h"

#include <ATen/CPUGeneratorImpl.h>

namespace echo_npu {

bool EchoNpuHooksInterface::hasPrimaryContext(c10::DeviceIndex device_index) const {
    int count = 0;
    echoNpuGetDeviceCount(&count);
    return device_index >= 0 && device_index < count;
}

const at::Generator& EchoNpuHooksInterface::getDefaultGenerator(c10::DeviceIndex device_index) const {
    // 使用CPU生成器作为默认实现
    static std::vector<at::Generator> default_gens;
    
    if (default_gens.empty()) {
        int count = 0;
        echoNpuGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) {
            default_gens.push_back(at::make_generator<at::CPUGeneratorImpl>());
        }
    }
    
    return default_gens.at(device_index);
}

// 注册Hooks
static bool hooks_registered [[maybe_unused]] = []() {
    at::RegisterPrivateUse1HooksInterface(new EchoNpuHooksInterface());
    return true;
}();

} // namespace echo_npu


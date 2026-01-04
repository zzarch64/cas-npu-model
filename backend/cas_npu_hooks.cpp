// CAS-NPU Hooks Implementation
#include "cas_npu_hooks.h"
#include "runtime/cas_npu_runtime.h"

#include <ATen/CPUGeneratorImpl.h>

namespace cas_npu {

bool CasNpuHooksInterface::hasPrimaryContext(c10::DeviceIndex device_index) const {
    int count = 0;
    casNpuGetDeviceCount(&count);
    return device_index >= 0 && device_index < count;
}

const at::Generator& CasNpuHooksInterface::getDefaultGenerator(c10::DeviceIndex device_index) const {
    // 使用CPU生成器作为默认实现
    static std::vector<at::Generator> default_gens;
    
    if (default_gens.empty()) {
        int count = 0;
        casNpuGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) {
            default_gens.push_back(at::make_generator<at::CPUGeneratorImpl>());
        }
    }
    
    return default_gens.at(device_index);
}

// 注册Hooks
static bool hooks_registered [[maybe_unused]] = []() {
    at::RegisterPrivateUse1HooksInterface(new CasNpuHooksInterface());
    return true;
}();

} // namespace cas_npu


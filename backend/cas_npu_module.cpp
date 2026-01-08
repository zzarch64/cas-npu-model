// CAS-NPU Python Module - Pybind11绑定
#include "runtime/cas_npu_runtime.h"
#include "runtime/cas_npu_debug.h"

#include <torch/extension.h>
#include <c10/core/Device.h>

namespace cas_npu {

// 获取CAS-NPU设备对象
c10::Device get_cas_npu_device(int device_index = 0) {
    return c10::Device(c10::DeviceType::PrivateUse1, device_index);
}

// 检查设备是否可用
bool is_available() {
    int count = 0;
    auto err = casNpuGetDeviceCount(&count);
    return err == CAS_NPU_SUCCESS && count > 0;
}

// 获取设备数量
int device_count() {
    int count = 0;
    casNpuGetDeviceCount(&count);
    return count;
}

// 获取当前设备索引
int current_device() {
    int device = 0;
    casNpuGetDevice(&device);
    return device;
}

// 设置当前设备
void set_device(int device) {
    auto err = casNpuSetDevice(device);
    TORCH_CHECK(err == CAS_NPU_SUCCESS, 
                "Failed to set device: ", casNpuGetErrorString(err));
}

// 同步设备
void synchronize(int device = -1) {
    if (device >= 0) {
        int old_device = 0;
        casNpuGetDevice(&old_device);
        casNpuSetDevice(device);
        casNpuDeviceSynchronize();
        casNpuSetDevice(old_device);
    } else {
        casNpuDeviceSynchronize();
    }
}

// 打印调试统计摘要
void print_debug_summary() {
    CAS_NPU_DEBUG_SUMMARY();
}

// Python模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CAS-NPU custom device extension for PyTorch";
    
    m.def("get_device", &get_cas_npu_device, 
          py::arg("device_index") = 0,
          "Get CAS-NPU device object");
    
    m.def("is_available", &is_available,
          "Check if CAS-NPU device is available");
    
    m.def("device_count", &device_count,
          "Get number of CAS-NPU devices");
    
    m.def("current_device", &current_device,
          "Get current device index");
    
    m.def("set_device", &set_device,
          py::arg("device"),
          "Set current device");
    
    m.def("synchronize", &synchronize,
          py::arg("device") = -1,
          "Synchronize device");
    
    m.def("print_debug_summary", &print_debug_summary,
          "Print CAS-NPU debug statistics summary");
}

} // namespace cas_npu


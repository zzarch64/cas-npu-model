// ECHO-NPU Python Module - Pybind11绑定
#include "runtime/echo_npu_runtime.h"
#include "runtime/echo_npu_debug.h"

#include <torch/extension.h>
#include <c10/core/Device.h>

namespace echo_npu {

// 获取ECHO-NPU设备对象
c10::Device get_echo_npu_device(int device_index = 0) {
    return c10::Device(c10::DeviceType::PrivateUse1, device_index);
}

// 检查设备是否可用
bool is_available() {
    int count = 0;
    auto err = echoNpuGetDeviceCount(&count);
    return err == ECHO_NPU_SUCCESS && count > 0;
}

// 获取设备数量
int device_count() {
    int count = 0;
    echoNpuGetDeviceCount(&count);
    return count;
}

// 获取当前设备索引
int current_device() {
    int device = 0;
    echoNpuGetDevice(&device);
    return device;
}

// 设置当前设备
void set_device(int device) {
    auto err = echoNpuSetDevice(device);
    TORCH_CHECK(err == ECHO_NPU_SUCCESS, 
                "Failed to set device: ", echoNpuGetErrorString(err));
}

// 同步设备
void synchronize(int device = -1) {
    if (device >= 0) {
        int old_device = 0;
        echoNpuGetDevice(&old_device);
        echoNpuSetDevice(device);
        echoNpuDeviceSynchronize();
        echoNpuSetDevice(old_device);
    } else {
        echoNpuDeviceSynchronize();
    }
}

// 打印调试统计摘要
void print_debug_summary() {
    ECHO_NPU_DEBUG_SUMMARY();
}

// Python模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ECHO-NPU custom device extension for PyTorch";
    
    m.def("get_device", &get_echo_npu_device, 
          py::arg("device_index") = 0,
          "Get ECHO-NPU device object");
    
    m.def("is_available", &is_available,
          "Check if ECHO-NPU device is available");
    
    m.def("device_count", &device_count,
          "Get number of ECHO-NPU devices");
    
    m.def("current_device", &current_device,
          "Get current device index");
    
    m.def("set_device", &set_device,
          py::arg("device"),
          "Set current device");
    
    m.def("synchronize", &synchronize,
          py::arg("device") = -1,
          "Synchronize device");
    
    m.def("print_debug_summary", &print_debug_summary,
          "Print ECHO-NPU debug statistics summary");
}

} // namespace echo_npu


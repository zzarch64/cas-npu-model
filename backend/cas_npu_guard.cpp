// CAS-NPU DeviceGuard Implementation
#include "cas_npu_guard.h"
#include "runtime/cas_npu_runtime.h"

#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>

#include <array>

namespace cas_npu {

namespace {

// 线程本地当前流（每个设备一个）
static thread_local std::array<c10::Stream, CAS_NPU_DEVICE_COUNT> current_streams = {
    c10::Stream::unpack3(0, 0, c10::DeviceType::PrivateUse1)
};

// 默认流
static std::array<c10::Stream, CAS_NPU_DEVICE_COUNT> default_streams = {
    c10::Stream::unpack3(0, 0, c10::DeviceType::PrivateUse1)
};

// 流ID生成器
static int64_t stream_id_gen = 1;

// 事件ID生成器
static int64_t event_id_gen = 1;

} // anonymous namespace

c10::DeviceType CasNpuGuardImpl::type() const {
    return static_type;
}

c10::Device CasNpuGuardImpl::exchangeDevice(c10::Device d) const {
    TORCH_CHECK(d.is_privateuseone(), 
                "Expected a PrivateUse1 device, but got ", d);
    
    int old_device = 0;
    auto err = casNpuGetDevice(&old_device);
    TORCH_CHECK(err == CAS_NPU_SUCCESS, 
                "Failed to get device: ", casNpuGetErrorString(err));
    
    err = casNpuSetDevice(d.index());
    TORCH_CHECK(err == CAS_NPU_SUCCESS, 
                "Failed to set device: ", casNpuGetErrorString(err));
    
    return c10::Device(static_type, old_device);
}

c10::Device CasNpuGuardImpl::getDevice() const {
    int device = 0;
    auto err = casNpuGetDevice(&device);
    TORCH_CHECK(err == CAS_NPU_SUCCESS, 
                "Failed to get device: ", casNpuGetErrorString(err));
    return c10::Device(static_type, device);
}

void CasNpuGuardImpl::setDevice(c10::Device d) const {
    TORCH_CHECK(d.is_privateuseone(), 
                "Expected a PrivateUse1 device, but got ", d);
    auto err = casNpuSetDevice(d.index());
    TORCH_CHECK(err == CAS_NPU_SUCCESS, 
                "Failed to set device: ", casNpuGetErrorString(err));
}

void CasNpuGuardImpl::uncheckedSetDevice(c10::Device d) const noexcept {
    casNpuSetDevice(d.index());
}

c10::DeviceIndex CasNpuGuardImpl::deviceCount() const noexcept {
    int count = 0;
    casNpuGetDeviceCount(&count);
    return static_cast<c10::DeviceIndex>(count);
}

c10::Stream CasNpuGuardImpl::getStream(c10::Device d) const noexcept {
    return current_streams[d.index()];
}

c10::Stream CasNpuGuardImpl::getDefaultStream(c10::Device d) const {
    return default_streams[d.index()];
}

c10::Stream CasNpuGuardImpl::getNewStream(c10::Device d, int priority) const {
    (void)priority;
    return c10::Stream::unpack3(stream_id_gen++, d.index(), d.type());
}

c10::Stream CasNpuGuardImpl::getStreamFromGlobalPool(c10::Device d, bool isHighPriority) const {
    return getNewStream(d, isHighPriority ? -1 : 0);
}

c10::Stream CasNpuGuardImpl::exchangeStream(c10::Stream s) const noexcept {
    auto old_stream = current_streams[s.device().index()];
    current_streams[s.device().index()] = s;
    return old_stream;
}

bool CasNpuGuardImpl::queryStream(const c10::Stream& stream) const {
    (void)stream;
    return true;  // CPU模拟，总是完成
}

void CasNpuGuardImpl::synchronizeStream(const c10::Stream& stream) const {
    (void)stream;
    casNpuDeviceSynchronize();
}

void CasNpuGuardImpl::synchronizeDevice(const c10::DeviceIndex device_index) const {
    (void)device_index;
    casNpuDeviceSynchronize();
}

void CasNpuGuardImpl::destroyEvent(void* event, const c10::DeviceIndex device_index) const noexcept {
    (void)event;
    (void)device_index;
    // 简单模拟，无需实际销毁
}

void CasNpuGuardImpl::record(void** event, const c10::Stream& stream,
                             const c10::DeviceIndex device_index,
                             const c10::EventFlag flag) const {
    TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
                "Event device index ", device_index,
                " does not match recording stream's device index ",
                stream.device_index(), ".");
    
    (void)flag;
    
    if (*event == nullptr) {
        *event = reinterpret_cast<void*>(event_id_gen++);
    }
}

void CasNpuGuardImpl::block(void* event, const c10::Stream& stream) const {
    (void)event;
    (void)stream;
    // CPU模拟，无需阻塞
}

bool CasNpuGuardImpl::queryEvent(void* event) const {
    (void)event;
    return true;  // CPU模拟，事件总是完成
}

void CasNpuGuardImpl::synchronizeEvent(void* event) const {
    (void)event;
    // CPU模拟，无需同步
}

double CasNpuGuardImpl::elapsedTime(void* event1, void* event2,
                                    const c10::DeviceIndex device_index) const {
    (void)event1;
    (void)event2;
    (void)device_index;
    return 0.0;  // CPU模拟，返回0
}

// 注册DeviceGuard到PrivateUse1
C10_REGISTER_GUARD_IMPL(PrivateUse1, CasNpuGuardImpl);

} // namespace cas_npu


// CAS-NPU Debug Header - 调试打印工具
// 用于跟踪算子执行类型和数据传输方向
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <atomic>

namespace cas_npu {
namespace debug {

// ============ 算子类型定义 ============
enum class OpType {
    NPU_NATIVE,      // NPU原生实现 (直接在设备执行,无CPU往返)
    CPU_FALLBACK,    // CPU Fallback (Device→CPU→计算→Device,有显式copy)
    VIEW_OP,         // View操作 (仅修改metadata,无数据拷贝)
    PURE_FALLBACK,   // 纯CPU Fallback (PyTorch native cpu_fallback)
    DATA_COPY        // 数据拷贝操作 (H↔D传输)
};

// ============ 数据传输方向 ============
enum class TransferDir {
    NONE,              // 无传输
    HOST_TO_DEVICE,    // CPU → Device
    DEVICE_TO_HOST,    // Device → CPU
    DEVICE_TO_DEVICE,  // Device → Device
    HOST_TO_HOST       // CPU → CPU
};

// ============ 全局调试开关 ============
inline bool is_debug_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("CAS_NPU_DEBUG");
        enabled = (env != nullptr && (std::string(env) == "1" || std::string(env) == "true"));
    }
    return enabled > 0;
}

// 详细程度级别: 0=关闭, 1=算子, 2=算子+传输, 3=全部(含统计)
inline int debug_level() {
    static int level = -1;
    if (level < 0) {
        const char* env = std::getenv("CAS_NPU_DEBUG_LEVEL");
        if (env != nullptr) {
            level = std::atoi(env);
        } else if (is_debug_enabled()) {
            level = 2;  // 默认显示算子和传输
        } else {
            level = 0;
        }
    }
    return level;
}

// ============ 颜色定义 (ANSI) - 已禁用，输出为纯文本日志 ============
// 如需恢复颜色输出，取消下面的注释
/*
namespace color {
    constexpr const char* RESET   = "\033[0m";
    constexpr const char* GREEN   = "\033[32m";  // NPU原生
    constexpr const char* YELLOW  = "\033[33m";  // CPU Fallback with copy
    constexpr const char* CYAN    = "\033[36m";  // View操作
    constexpr const char* RED     = "\033[31m";  // 纯CPU Fallback
    constexpr const char* BLUE    = "\033[34m";  // 数据传输
    constexpr const char* MAGENTA = "\033[35m";  // Runtime
    constexpr const char* BOLD    = "\033[1m";
    constexpr const char* DIM     = "\033[2m";
}
*/

// ============ 统计计数器 ============
struct DebugStats {
    std::atomic<uint64_t> npu_native_ops{0};
    std::atomic<uint64_t> cpu_fallback_ops{0};
    std::atomic<uint64_t> view_ops{0};
    std::atomic<uint64_t> pure_fallback_ops{0};
    std::atomic<uint64_t> h2d_transfers{0};
    std::atomic<uint64_t> d2h_transfers{0};
    std::atomic<uint64_t> d2d_transfers{0};
    std::atomic<uint64_t> h2d_bytes{0};
    std::atomic<uint64_t> d2h_bytes{0};
    std::atomic<uint64_t> d2d_bytes{0};
    
    void reset() {
        npu_native_ops = 0;
        cpu_fallback_ops = 0;
        view_ops = 0;
        pure_fallback_ops = 0;
        h2d_transfers = 0;
        d2h_transfers = 0;
        d2d_transfers = 0;
        h2d_bytes = 0;
        d2h_bytes = 0;
        d2d_bytes = 0;
    }
    
    void print_summary() {
        fprintf(stdout, "\n========== CAS-NPU Debug Summary ==========\n");
        fprintf(stdout, "[Operator Statistics]\n");
        fprintf(stdout, "  * NPU Native ops:      %lu\n", npu_native_ops.load());
        fprintf(stdout, "  * CPU Fallback ops:    %lu\n", cpu_fallback_ops.load());
        fprintf(stdout, "  * View ops:            %lu\n", view_ops.load());
        fprintf(stdout, "  * Pure Fallback ops:   %lu\n", pure_fallback_ops.load());
        fprintf(stdout, "\n[Data Transfer Statistics]\n");
        fprintf(stdout, "  ^ Host->Device:  %lu times, %.2f MB\n",
                h2d_transfers.load(), h2d_bytes.load() / 1024.0 / 1024.0);
        fprintf(stdout, "  v Device->Host:  %lu times, %.2f MB\n",
                d2h_transfers.load(), d2h_bytes.load() / 1024.0 / 1024.0);
        fprintf(stdout, "  <-> Device<->Device: %lu times, %.2f MB\n",
                d2d_transfers.load(), d2d_bytes.load() / 1024.0 / 1024.0);
        fprintf(stdout, "==============================================\n\n");
        fflush(stdout);
    }
};

inline DebugStats& get_stats() {
    static DebugStats stats;
    return stats;
}

// ============ 打印辅助函数 ============

inline const char* op_type_str(OpType type) {
    switch (type) {
        case OpType::NPU_NATIVE:   return "NPU";
        case OpType::CPU_FALLBACK: return "CPU←→NPU";
        case OpType::VIEW_OP:      return "VIEW";
        case OpType::PURE_FALLBACK:return "CPU";
        case OpType::DATA_COPY:    return "COPY";
        default:                   return "???";
    }
}

// 颜色函数已禁用，输出为纯文本日志
// inline const char* op_type_color(OpType type) { ... }

inline const char* transfer_dir_str(TransferDir dir) {
    switch (dir) {
        case TransferDir::HOST_TO_DEVICE:   return "H->D";
        case TransferDir::DEVICE_TO_HOST:   return "D->H";
        case TransferDir::DEVICE_TO_DEVICE: return "D->D";
        case TransferDir::HOST_TO_HOST:     return "H->H";
        case TransferDir::NONE:
        default:                            return "";
    }
}

// ============ 主要调试宏 ============

// 打印算子执行信息
#define CAS_NPU_DEBUG_OP(op_type, op_name, fmt, ...) \
    do { \
        if (::cas_npu::debug::debug_level() >= 1) { \
            auto& stats = ::cas_npu::debug::get_stats(); \
            switch (op_type) { \
                case ::cas_npu::debug::OpType::NPU_NATIVE: stats.npu_native_ops++; break; \
                case ::cas_npu::debug::OpType::CPU_FALLBACK: stats.cpu_fallback_ops++; break; \
                case ::cas_npu::debug::OpType::VIEW_OP: stats.view_ops++; break; \
                case ::cas_npu::debug::OpType::PURE_FALLBACK: stats.pure_fallback_ops++; break; \
                case ::cas_npu::debug::OpType::DATA_COPY: break; /* 传输统计单独处理 */ \
            } \
            fprintf(stdout, "[%s] %s" fmt "\n", \
                    ::cas_npu::debug::op_type_str(op_type), \
                    op_name, \
                    ##__VA_ARGS__); \
            fflush(stdout); \
        } \
    } while(0)

// 打印数据传输信息
#define CAS_NPU_DEBUG_TRANSFER(dir, bytes, fmt, ...) \
    do { \
        if (::cas_npu::debug::debug_level() >= 2) { \
            auto& stats = ::cas_npu::debug::get_stats(); \
            switch (dir) { \
                case ::cas_npu::debug::TransferDir::HOST_TO_DEVICE: \
                    stats.h2d_transfers++; stats.h2d_bytes += bytes; break; \
                case ::cas_npu::debug::TransferDir::DEVICE_TO_HOST: \
                    stats.d2h_transfers++; stats.d2h_bytes += bytes; break; \
                case ::cas_npu::debug::TransferDir::DEVICE_TO_DEVICE: \
                    stats.d2d_transfers++; stats.d2d_bytes += bytes; break; \
                default: break; \
            } \
            fprintf(stdout, "  [COPY] _copy_from %s %.2f KB" fmt "\n", \
                    ::cas_npu::debug::transfer_dir_str(dir), \
                    static_cast<double>(bytes) / 1024.0, \
                    ##__VA_ARGS__); \
            fflush(stdout); \
        } \
    } while(0)

// 打印Runtime层信息
#define CAS_NPU_DEBUG_RUNTIME(api_name, fmt, ...) \
    do { \
        if (::cas_npu::debug::debug_level() >= 3) { \
            fprintf(stdout, "      Runtime: %s" fmt "\n", \
                    api_name, \
                    ##__VA_ARGS__); \
            fflush(stdout); \
        } \
    } while(0)

// 打印统计摘要
#define CAS_NPU_DEBUG_SUMMARY() \
    do { \
        if (::cas_npu::debug::debug_level() >= 1) { \
            ::cas_npu::debug::get_stats().print_summary(); \
        } \
    } while(0)

// 重置统计
#define CAS_NPU_DEBUG_RESET() \
    do { \
        ::cas_npu::debug::get_stats().reset(); \
    } while(0)

} // namespace debug
} // namespace cas_npu

# ECHO-NPU Python Package
# 自定义设备扩展，使用PrivateUse1 dispatch key

import sys
import torch
from torch.utils.backend_registration import (
    rename_privateuse1_backend,
    generate_methods_for_privateuse1_backend,
)

# 设置 stdout 为行缓冲模式，确保与 C++ 调试输出的顺序正确
# 当重定向到文件时，Python 默认使用全缓冲，会导致输出顺序混乱
# 这确保 Python print() 和 C++ fprintf(stdout) 的输出顺序一致
# 注意：只有在启用调试时才设置，避免影响正常使用
import os
_debug_enabled = os.environ.get('ECHO_NPU_DEBUG', '0') in ('1', 'true')
_debug_level = int(os.environ.get('ECHO_NPU_DEBUG_LEVEL', '0'))
if (_debug_enabled or _debug_level > 0) and not sys.stdout.isatty():
    # 只有在启用调试且重定向到文件时才设置（终端默认就是行缓冲）
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            # Python 3.7+ 支持 reconfigure
            sys.stdout.reconfigure(line_buffering=True)
        # Python < 3.7 需要用户使用 python -u 运行，或手动 flush
    except Exception:
        # 如果设置失败，忽略（不影响功能）
        pass

# 重命名PrivateUse1后端为echo_npu
_BACKEND_NAME = "echo_npu"

def _init_extension():
    """初始化ECHO-NPU扩展"""
    try:
        # 导入编译的C++扩展
        from . import _echo_npu_C as _C
        
        # 重命名后端
        rename_privateuse1_backend(_BACKEND_NAME)
        
        # 生成Tensor和Module的便捷方法
        generate_methods_for_privateuse1_backend(
            for_tensor=True,
            for_module=True,
            for_storage=True,
            for_packed_sequence=True,
        )
        
        # 注册设备模块
        torch._register_device_module(_BACKEND_NAME, _EchoNpuModule())
        
        return _C
    except ImportError as e:
        raise ImportError(
            f"Failed to import ECHO-NPU extension. "
            f"Make sure the extension is properly built. Error: {e}"
        )

class _EchoNpuModule:
    """ECHO-NPU设备模块，提供设备相关的API"""
    
    def __init__(self):
        from . import _echo_npu_C as _C
        self._C = _C
    
    def is_available(self) -> bool:
        """检查ECHO-NPU设备是否可用"""
        return self._C.is_available()
    
    def device_count(self) -> int:
        """获取设备数量"""
        return self._C.device_count()
    
    def current_device(self) -> int:
        """获取当前设备索引"""
        return self._C.current_device()
    
    def set_device(self, device: int) -> None:
        """设置当前设备"""
        self._C.set_device(device)
    
    def synchronize(self, device: int = -1) -> None:
        """同步设备"""
        self._C.synchronize(device)
    
    def get_device(self, device_index: int = 0) -> torch.device:
        """获取设备对象"""
        return self._C.get_device(device_index)
    
    # 以下是AMP相关的方法
    def get_amp_supported_dtype(self):
        """返回AMP支持的数据类型"""
        return [torch.float16, torch.bfloat16]
    
    # 随机数相关
    def _is_in_bad_fork(self) -> bool:
        return False
    
    def manual_seed_all(self, seed: int) -> None:
        pass
    
    def get_rng_state(self, device=None):
        return torch.empty(0, dtype=torch.uint8)
    
    def set_rng_state(self, state, device=None) -> None:
        pass

# 自动初始化
_C = _init_extension()

# 在程序退出时自动打印调试统计摘要（如果启用了调试）
import atexit
def _print_debug_summary_on_exit():
    """程序退出时打印调试统计摘要"""
    try:
        # 检查是否启用了调试
        import os
        debug_enabled = os.environ.get('ECHO_NPU_DEBUG', '0') in ('1', 'true')
        debug_level = int(os.environ.get('ECHO_NPU_DEBUG_LEVEL', '0'))
        if debug_enabled and debug_level >= 1:
            _C.print_debug_summary()
    except:
        pass  # 忽略错误，避免影响程序正常退出

atexit.register(_print_debug_summary_on_exit)

# 导出公共API
def is_available() -> bool:
    """检查ECHO-NPU设备是否可用"""
    return _C.is_available()

def device_count() -> int:
    """获取设备数量"""
    return _C.device_count()

def current_device() -> int:
    """获取当前设备索引"""
    return _C.current_device()

def set_device(device: int) -> None:
    """设置当前设备"""
    _C.set_device(device)

def synchronize(device: int = -1) -> None:
    """同步设备"""
    _C.synchronize(device)

def get_device(device_index: int = 0) -> torch.device:
    """获取设备对象"""
    return _C.get_device(device_index)

__all__ = [
    'is_available',
    'device_count', 
    'current_device',
    'set_device',
    'synchronize',
    'get_device',
]


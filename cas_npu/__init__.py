# CAS-NPU Python Package
# 自定义设备扩展，使用PrivateUse1 dispatch key

import torch
from torch.utils.backend_registration import (
    rename_privateuse1_backend,
    generate_methods_for_privateuse1_backend,
)

# 重命名PrivateUse1后端为cas_npu
_BACKEND_NAME = "cas_npu"

def _init_extension():
    """初始化CAS-NPU扩展"""
    try:
        # 导入编译的C++扩展
        from . import _cas_npu_C as _C
        
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
        torch._register_device_module(_BACKEND_NAME, _CasNpuModule())
        
        return _C
    except ImportError as e:
        raise ImportError(
            f"Failed to import CAS-NPU extension. "
            f"Make sure the extension is properly built. Error: {e}"
        )

class _CasNpuModule:
    """CAS-NPU设备模块，提供设备相关的API"""
    
    def __init__(self):
        from . import _cas_npu_C as _C
        self._C = _C
    
    def is_available(self) -> bool:
        """检查CAS-NPU设备是否可用"""
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

# 导出公共API
def is_available() -> bool:
    """检查CAS-NPU设备是否可用"""
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


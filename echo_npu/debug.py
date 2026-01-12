# ECHO-NPU Debug Module
# 用于控制调试打印的Python接口
"""
调试打印使用说明：

1. 通过环境变量启用调试打印：
   - ECHO_NPU_DEBUG=1          启用基本调试打印
   - ECHO_NPU_DEBUG_LEVEL=N    设置详细程度 (1-3)
     - Level 1: 仅显示算子执行信息
     - Level 2: 显示算子执行 + 数据传输 (默认)
     - Level 3: 全部信息 (含Runtime层)

2. 在Python中使用：
   import echo_npu.debug as debug
   debug.enable()              # 启用调试
   debug.disable()             # 禁用调试
   debug.set_level(3)          # 设置详细程度

3. 图例说明：
   [NPU]      绿色 - NPU原生实现 (高性能,无CPU往返)
   [CPU←→NPU] 黄色 - 显式CPU Fallback (Device→CPU→计算→Device)
   [VIEW]     青色 - View操作 (仅修改metadata,无数据移动)
   [CPU]      红色 - 纯CPU Fallback (通过PyTorch cpu_fallback)
   [COPY]     蓝色 - 数据拷贝操作 (H↔D传输,如.to(device))
   
   数据传输:
   [H→D] - Host到Device传输
   [D→H] - Device到Host传输
   [D→D] - Device到Device传输

4. 示例输出：
   [NPU] mm [128x768] @ [768x3072]
   [CPU←→NPU] rsqrt [98304]
       ↳ [D→H] 384.00 KB
       ↳ [H→D] 384.00 KB
   [VIEW] reshape
   [CPU] aten::embedding
       ↳ [D→H] 384.00 KB
       ↳ [H→D] 384.00 KB
"""

import os

def enable(level: int = 2):
    """
    启用调试打印
    
    Args:
        level: 调试详细程度 (1-3)
            1 = 仅算子执行
            2 = 算子 + 数据传输 (默认)
            3 = 全部信息
    """
    os.environ['ECHO_NPU_DEBUG'] = '1'
    os.environ['ECHO_NPU_DEBUG_LEVEL'] = str(level)
    print(f"[ECHO-NPU Debug] 已启用调试打印, Level={level}")
    print("  图例: [NPU]=原生  [CPU←→NPU]=显式Fallback  [VIEW]=View  [CPU]=纯Fallback  [COPY]=传输")

def disable():
    """禁用调试打印"""
    os.environ['ECHO_NPU_DEBUG'] = '0'
    os.environ['ECHO_NPU_DEBUG_LEVEL'] = '0'
    print("[ECHO-NPU Debug] 已禁用调试打印")

def set_level(level: int):
    """
    设置调试详细程度
    
    Args:
        level: 1=算子, 2=算子+传输, 3=全部
    """
    os.environ['ECHO_NPU_DEBUG_LEVEL'] = str(level)
    print(f"[ECHO-NPU Debug] 详细程度设置为 Level={level}")

def is_enabled() -> bool:
    """检查调试打印是否已启用"""
    env = os.environ.get('ECHO_NPU_DEBUG', '0')
    return env == '1' or env.lower() == 'true'

def get_level() -> int:
    """获取当前调试详细程度"""
    return int(os.environ.get('ECHO_NPU_DEBUG_LEVEL', '0'))

# Context manager for temporary debug mode
class debug_mode:
    """
    临时启用调试模式的上下文管理器
    
    Example:
        with debug_mode(level=2):
            model(input)  # 这里的操作会打印调试信息
    """
    def __init__(self, level: int = 2):
        self.level = level
        self.old_debug = None
        self.old_level = None
    
    def __enter__(self):
        self.old_debug = os.environ.get('ECHO_NPU_DEBUG', '0')
        self.old_level = os.environ.get('ECHO_NPU_DEBUG_LEVEL', '0')
        os.environ['ECHO_NPU_DEBUG'] = '1'
        os.environ['ECHO_NPU_DEBUG_LEVEL'] = str(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.environ['ECHO_NPU_DEBUG'] = self.old_debug
        os.environ['ECHO_NPU_DEBUG_LEVEL'] = self.old_level
        return False

def print_summary():
    """
    打印调试统计摘要
    
    显示算子执行统计和数据传输统计。
    如果调试未启用，此函数不会打印任何内容。
    """
    try:
        import echo_npu
        echo_npu._C.print_debug_summary()
    except ImportError:
        print("[ECHO-NPU Debug] 无法打印统计摘要：扩展未加载")
    except AttributeError:
        print("[ECHO-NPU Debug] 无法打印统计摘要：接口不可用")

__all__ = [
    'enable',
    'disable', 
    'set_level',
    'is_enabled',
    'get_level',
    'debug_mode',
    'print_summary',
]

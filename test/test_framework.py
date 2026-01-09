#!/usr/bin/env python
"""
CAS-NPU 测试框架

提供统一的测试工具函数、参数配置和结果报告功能。
"""

import sys
import os
import argparse
import torch
from typing import Optional, Tuple, Dict, Any
from enum import Enum

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cas_npu
    CAS_NPU_AVAILABLE = True
except ImportError as e:
    CAS_NPU_AVAILABLE = False
    CAS_NPU_IMPORT_ERROR = str(e)


class VerbosityLevel(Enum):
    """详细程度级别"""
    QUIET = 0      # 只显示结果
    NORMAL = 1     # 正常输出
    VERBOSE = 2    # 详细输出
    DEBUG = 3      # 调试输出（包含所有中间步骤）


class TestConfig:
    """测试配置类"""
    def __init__(self):
        self.verbosity = VerbosityLevel.NORMAL
        self.device = 'cas_npu:0'
        self.check_nan = True
        self.check_inf = True
        self.check_gradient = True
        self.tolerance = 1e-5
        self.show_stats = True
        self.show_shape = True
        self.show_device = True
        self.max_nan_positions = 10  # 最多显示的NaN位置数
        
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TestConfig':
        """从命令行参数创建配置"""
        config = cls()
        if hasattr(args, 'verbose'):
            if args.verbose >= 3:
                config.verbosity = VerbosityLevel.DEBUG
            elif args.verbose >= 2:
                config.verbosity = VerbosityLevel.VERBOSE
            elif args.verbose >= 1:
                config.verbosity = VerbosityLevel.NORMAL
            else:
                config.verbosity = VerbosityLevel.QUIET
        
        if hasattr(args, 'device'):
            config.device = args.device
        
        if hasattr(args, 'tolerance'):
            config.tolerance = args.tolerance
            
        if hasattr(args, 'quiet'):
            if args.quiet:
                config.verbosity = VerbosityLevel.QUIET
                
        return config


def ensure_cas_npu():
    """确保CAS-NPU扩展已导入"""
    if not CAS_NPU_AVAILABLE:
        print(f"✗ Failed to import CAS-NPU extension: {CAS_NPU_IMPORT_ERROR}")
        sys.exit(1)
    return True


def check_tensor(
    tensor: Optional[torch.Tensor],
    name: str,
    config: Optional[TestConfig] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    检查tensor的详细信息
    
    Args:
        tensor: 要检查的tensor
        name: tensor的名称
        config: 测试配置
        device: 可选的设备（用于转换）
    
    Returns:
        包含检查结果的字典
    """
    if config is None:
        config = TestConfig()
    
    result = {
        'name': name,
        'is_none': tensor is None,
        'has_nan': False,
        'has_inf': False,
        'nan_count': 0,
        'inf_count': 0,
        'total_count': 0,
        'shape': None,
        'device': None,
        'dtype': None,
        'stats': {}
    }
    
    if tensor is None:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  {name}: None")
        return result
    
    # 转换到CPU进行检查
    if device is not None:
        cpu_tensor = tensor.cpu() if tensor.device != torch.device('cpu') else tensor
    else:
        cpu_tensor = tensor.cpu() if hasattr(tensor, 'device') and tensor.device.type == 'cas_npu' else tensor
    
    result['shape'] = tuple(tensor.shape)
    result['device'] = str(tensor.device)
    result['dtype'] = str(tensor.dtype)
    result['total_count'] = cpu_tensor.numel()
    
    # 检查NaN和Inf
    has_nan = torch.isnan(cpu_tensor).any().item()
    has_inf = torch.isinf(cpu_tensor).any().item()
    
    result['has_nan'] = has_nan
    result['has_inf'] = has_inf
    
    if has_nan:
        result['nan_count'] = torch.isnan(cpu_tensor).sum().item()
    if has_inf:
        result['inf_count'] = torch.isinf(cpu_tensor).sum().item()
    
    # 计算统计信息
    if not has_nan and not has_inf:
        result['stats'] = {
            'min': cpu_tensor.min().item(),
            'max': cpu_tensor.max().item(),
            'mean': cpu_tensor.mean().item(),
            'std': cpu_tensor.std().item()
        }
    
    # 根据详细程度输出
    if config.verbosity == VerbosityLevel.QUIET:
        if has_nan or has_inf:
            print(f"✗ {name}: NaN={has_nan}, Inf={has_inf}")
    elif config.verbosity == VerbosityLevel.NORMAL:
        status = "✗ NaN/Inf" if (has_nan or has_inf) else "✓ OK"
        info_parts = []
        if config.show_shape:
            info_parts.append(f"shape={result['shape']}")
        if config.show_device:
            info_parts.append(f"device={result['device']}")
        info_str = ", ".join(info_parts) if info_parts else ""
        print(f"  {name}: {status}" + (f", {info_str}" if info_str else ""))
        
        if has_nan:
            print(f"    NaN: {result['nan_count']}/{result['total_count']} ({result['nan_count']/result['total_count']*100:.2f}%)")
        if has_inf:
            print(f"    Inf: {result['inf_count']}/{result['total_count']} ({result['inf_count']/result['total_count']*100:.2f}%)")
        if not has_nan and not has_inf and config.show_stats:
            print(f"    min={result['stats']['min']:.6f}, max={result['stats']['max']:.6f}, mean={result['stats']['mean']:.6f}")
    else:  # VERBOSE or DEBUG
        print(f"\n=== Checking {name} ===")
        print(f"  Shape: {result['shape']}")
        print(f"  Device: {result['device']}")
        print(f"  Dtype: {result['dtype']}")
        
        if has_nan:
            nan_count = result['nan_count']
            print(f"  ✗ Contains NaN: {nan_count}/{result['total_count']} ({nan_count/result['total_count']*100:.2f}%)")
            if config.verbosity == VerbosityLevel.DEBUG:
                nan_mask = torch.isnan(cpu_tensor)
                nan_indices = torch.nonzero(nan_mask, as_tuple=False)
                if len(nan_indices) > 0:
                    max_show = min(config.max_nan_positions, len(nan_indices))
                    print(f"    First {max_show} NaN positions:")
                    for i, idx in enumerate(nan_indices[:max_show]):
                        print(f"      {idx.tolist()}")
        else:
            print(f"  ✓ No NaN")
        
        if has_inf:
            inf_count = result['inf_count']
            print(f"  ✗ Contains Inf: {inf_count}/{result['total_count']} ({inf_count/result['total_count']*100:.2f}%)")
        else:
            print(f"  ✓ No Inf")
        
        if not has_nan and not has_inf:
            stats = result['stats']
            print(f"  Stats: min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    
    return result


def verify_tensor_match(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str = "tensor",
    tolerance: float = 1e-5,
    config: Optional[TestConfig] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    验证两个tensor是否匹配
    
    Returns:
        (是否匹配, 详细信息字典)
    """
    if config is None:
        config = TestConfig()
    
    result = {
        'matched': False,
        'max_diff': None,
        'mean_diff': None,
        'shape_match': False
    }
    
    actual_cpu = actual.cpu() if actual.device.type == 'cas_npu' else actual
    expected_cpu = expected.cpu() if expected.device.type == 'cas_npu' else expected
    
    # 检查形状
    if actual_cpu.shape != expected_cpu.shape:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ {name}: Shape mismatch! Expected {expected_cpu.shape}, got {actual_cpu.shape}")
        return False, result
    
    result['shape_match'] = True
    
    # 检查NaN
    actual_has_nan = torch.isnan(actual_cpu).any().item()
    expected_has_nan = torch.isnan(expected_cpu).any().item()
    
    if actual_has_nan or expected_has_nan:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"  ✗ {name}: Contains NaN, cannot compare")
        return False, result
    
    # 计算差异
    diff = (actual_cpu - expected_cpu).abs()
    result['max_diff'] = diff.max().item()
    result['mean_diff'] = diff.mean().item()
    result['matched'] = result['max_diff'] < tolerance
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        if result['matched']:
            print(f"  ✓ {name}: Matches expected (max_diff={result['max_diff']:.6f})")
        else:
            print(f"  ✗ {name}: Doesn't match (max_diff={result['max_diff']:.6f}, tolerance={tolerance})")
    
    return result['matched'], result


def analyze_nan_distribution(
    tensor: torch.Tensor,
    name: str = "tensor",
    config: Optional[TestConfig] = None
) -> Dict[str, Any]:
    """分析NaN的分布模式"""
    if config is None:
        config = TestConfig()
    
    cpu_tensor = tensor.cpu() if tensor.device.type == 'cas_npu' else tensor
    nan_mask = torch.isnan(cpu_tensor)
    
    result = {
        'total_nan': nan_mask.sum().item(),
        'total_elements': cpu_tensor.numel(),
        'nan_rows': 0,
        'nan_cols': 0,
        'nan_row_indices': [],
        'nan_col_indices': []
    }
    
    if result['total_nan'] == 0:
        return result
    
    # 按维度分析
    if len(cpu_tensor.shape) >= 2:
        nan_rows = nan_mask.any(dim=1)
        nan_cols = nan_mask.any(dim=0)
        result['nan_rows'] = nan_rows.sum().item()
        result['nan_cols'] = nan_cols.sum().item()
        
        if result['nan_rows'] > 0:
            result['nan_row_indices'] = torch.nonzero(nan_rows, as_tuple=False).squeeze().tolist()
            if not isinstance(result['nan_row_indices'], list):
                result['nan_row_indices'] = [result['nan_row_indices']]
        
        if result['nan_cols'] > 0:
            result['nan_col_indices'] = torch.nonzero(nan_cols, as_tuple=False).squeeze().tolist()
            if not isinstance(result['nan_col_indices'], list):
                result['nan_col_indices'] = [result['nan_col_indices']]
    
    if config.verbosity.value >= VerbosityLevel.VERBOSE.value:
        print(f"\n[NaN Distribution Analysis for {name}]")
        print(f"  Total NaN: {result['total_nan']}/{result['total_elements']}")
        if len(cpu_tensor.shape) >= 2:
            print(f"  NaN rows: {result['nan_rows']}/{cpu_tensor.shape[0]}")
            print(f"  NaN cols: {result['nan_cols']}/{cpu_tensor.shape[1]}")
            if result['nan_row_indices']:
                max_show = min(config.max_nan_positions, len(result['nan_row_indices']))
                print(f"  First {max_show} NaN row indices: {result['nan_row_indices'][:max_show]}")
    
    return result


def print_section(title: str, config: Optional[TestConfig] = None):
    """打印章节标题"""
    if config is None:
        config = TestConfig()
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)


def print_step(step_name: str, config: Optional[TestConfig] = None):
    """打印步骤标题"""
    if config is None:
        config = TestConfig()
    
    if config.verbosity.value >= VerbosityLevel.NORMAL.value:
        print(f"\n[{step_name}]")


def create_arg_parser(description: str = "CAS-NPU Test") -> argparse.ArgumentParser:
    """创建标准的命令行参数解析器"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=1,
        help='增加输出详细程度（-v: normal, -vv: verbose, -vvv: debug）'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='安静模式，只显示结果'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cas_npu:0',
        help='使用的设备（默认: cas_npu:0）'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-5,
        help='数值比较的容差（默认: 1e-5）'
    )
    parser.add_argument(
        '--no-check-nan',
        action='store_true',
        help='不检查NaN'
    )
    parser.add_argument(
        '--no-check-inf',
        action='store_true',
        help='不检查Inf'
    )
    return parser


def run_test(test_func, config: TestConfig, test_name: str = "Test") -> bool:
    """运行测试并处理异常"""
    try:
        print_section(f"{test_name}", config)
        result = test_func(config)
        return result
    except Exception as e:
        if config.verbosity.value >= VerbosityLevel.NORMAL.value:
            print(f"\n✗ {test_name} failed: {e}")
            if config.verbosity.value >= VerbosityLevel.DEBUG.value:
                import traceback
                traceback.print_exc()
        return False


if __name__ == "__main__":
    # 测试框架本身
    print("CAS-NPU Test Framework")
    print("=" * 80)
    
    ensure_cas_npu()
    print("✓ CAS-NPU extension imported successfully")
    
    # 测试配置
    parser = create_arg_parser("Test Framework")
    args = parser.parse_args()
    config = TestConfig.from_args(args)
    
    print(f"Verbosity: {config.verbosity.name}")
    print(f"Device: {config.device}")
    print(f"Tolerance: {config.tolerance}")

# NPU-CAS Extension Setup Script (Debug Version)
# 用于 gdb 调试，包含调试符号和禁用优化
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 选择模拟器：'cmodel' 或 'fpga'，默认使用 cmodel
impl_type = os.environ.get('CAS_NPU_IMPL', 'cmodel')
if impl_type not in ['cmodel', 'fpga']:
    impl_type = 'cmodel'
    print(f"Warning: Invalid cas_npu_IMPL value, using 'cmodel'")

print(f"Building DEBUG version with {impl_type} implementation")
print("  - Debug symbols: -g")
print("  - Optimization: -O0")
print("  - Warnings: -Wall -Wextra")

# 源文件列表
sources = [
    # Runtime API 层（PyTorch 集成）
    'backend/cas_npu_allocator.cpp',
    'backend/cas_npu_guard.cpp',
    'backend/cas_npu_hooks.cpp',
    'backend/cas_npu_ops.cpp',
    'backend/cas_npu_module.cpp',
    # 自定义算子示例
    'backend/cas_npu_custom_ops_example.cpp',
    # Runtime 实现层（cmodel 或 fpga simulator）
    f'runtime/{impl_type}/simulator.cpp',
]

# 头文件目录
project_root = os.path.dirname(__file__)
include_dirs = [
    project_root,
    os.path.join(project_root, 'backend'),
]

# 调试编译选项：-g 添加调试符号，-O0 禁用优化
extra_compile_args = {
    'cxx': ['-std=c++17', '-g', '-O0', '-Wall', '-Wextra', '-fno-omit-frame-pointer'],
}

setup(
    name='cas_npu',
    version='0.1.0',
    description='CAS-NPU custom device extension for PyTorch using PrivateUse1 (DEBUG BUILD)',
    author='CAS-NPU Team',
    packages=['cas_npu'],
    package_dir={'cas_npu': 'cas_npu'},
    ext_modules=[
        CppExtension(
            name='cas_npu._cas_npu_C',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.8',
)

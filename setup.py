# NPU-CAS Extension Setup Script
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 选择模拟器：'cmodel' 或 'fpga'，默认使用 cmodel
# 可以通过环境变量 CAS_NPU_IMPL 来指定
impl_type = os.environ.get('CAS_NPU_IMPL', 'cmodel')
if impl_type not in ['cmodel', 'fpga']:
    impl_type = 'cmodel'
    print(f"Warning: Invalid cas_npu_IMPL value, using 'cmodel'")

print(f"Building with {impl_type} implementation")

# 源文件列表
sources = [
    # Runtime API 层（PyTorch 集成）
    'backend/cas_npu_allocator.cpp',
    'backend/cas_npu_guard.cpp',
    'backend/cas_npu_hooks.cpp',
    'backend/cas_npu_ops.cpp',
    'backend/cas_npu_module.cpp',
    # Runtime 实现层（cmodel 或 fpga simulator）
    f'runtime/{impl_type}/simulator.cpp',
]

# 头文件目录
# 项目根目录需要包含，以便使用 "runtime/cas_npu_runtime.h" 这样的路径
project_root = os.path.dirname(__file__)
include_dirs = [
    project_root,  # 项目根目录，用于 "runtime/cas_npu_runtime.h"
    os.path.join(project_root, 'backend'),  # backend 目录，用于相对路径包含
]

# 编译选项
extra_compile_args = {
    'cxx': ['-std=c++17', '-O3', '-Wall'],
}

setup(
    name='cas_npu',
    version='0.1.0',
    description='CAS-NPU custom device extension for PyTorch using PrivateUse1',
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


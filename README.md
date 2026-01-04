# CAS-NPU Custom Device Extension for PyTorch

这是一个使用PyTorch的PrivateUse1机制实现的自定义设备扩展示例。该扩展模拟了一个名为"cas-npu"的NPU设备，并实现了基本的tensor操作，特别是`add.Tensor`。

## 目录结构

```
cas_npu_extension/
├── backend/                       # PyTorch API 层
│   ├── cas_npu_allocator.h/cpp    # 设备内存分配器
│   ├── cas_npu_guard.h/cpp        # DeviceGuard实现
│   ├── cas_npu_hooks.h/cpp        # PrivateUse1 Hooks
│   ├── cas_npu_ops.cpp            # 操作实现（add.Tensor等）
│   └── cas_npu_module.cpp         # Python绑定
├── runtime/                        # Runtime 层
│   ├── cas_npu_runtime.h          # 统一接口定义
│   ├── cmodel/                    # C 模型模拟器
│   │   └── simulator.cpp
│   └── fpga/                      # FPGA 模拟器
│       └── simulator.cpp
├── cas_npu/                       # Python包
│   └── __init__.py                # Python API
├── test/                          # 测试文件
│   ├── test_cas_npu.py
│   └── ...
├── setup.py                       # 构建脚本
└── README.md                      # 说明文档
```

## 实现原理

### 1. Runtime驱动层 (runtime/cas_npu_runtime.h, runtime/cmodel/simulator.cpp)

使用C++封装了一套模拟的NPU backend API：
- `casNpuMalloc/casNpuFree`: 内存分配/释放
- `casNpuMemcpy/casNpuMemset`: 内存操作
- `casNpuAddTensor`: 加法运算核心实现

实际上使用CPU来模拟NPU的行为。

### 2. 设备分配器 (backend/cas_npu_allocator.h/cpp)

实现`c10::Allocator`接口，用于在PrivateUse1设备上分配内存：
- 通过`REGISTER_ALLOCATOR`宏注册到PyTorch

### 3. DeviceGuard (backend/cas_npu_guard.h/cpp)

实现`c10::impl::DeviceGuardImplInterface`，管理设备切换和流：
- 通过`C10_REGISTER_GUARD_IMPL`宏注册

### 4. 操作注册 (backend/cas_npu_ops.cpp)

使用`TORCH_LIBRARY_IMPL`将操作注册到PrivateUse1 dispatch key：
- `add.Tensor`: 核心加法操作，调用runtime完成计算
- `empty.memory_format`, `empty_strided`: tensor创建
- `_copy_from`: 设备间数据拷贝
- 其他基础操作使用CPU fallback

### 5. Python绑定 (backend/cas_npu_module.cpp, python/__init__.py)

- C++模块通过pybind11暴露API
- Python层使用`rename_privateuse1_backend`重命名后端
- 使用`generate_methods_for_privateuse1_backend`生成便捷方法

## 构建和测试

### 前置要求

- PyTorch (已安装)
- C++17兼容的编译器
- Python 3.8+

### 构建

```bash
# 方法1: 使用脚本
chmod +x test/build_and_test.sh
./test/build_and_test.sh

# 方法2: 手动构建
python setup.py build_ext --inplace
```

### 测试

```bash
python test/test_cas_npu.py
```

## 使用示例

```python
import torch

# 导入并初始化扩展
import sys
sys.path.insert(0, '/path/to/cas_npu_extension')
from python import _cas_npu_C as cas_npu_C

# 设置后端
from torch.utils.backend_registration import (
    rename_privateuse1_backend,
    generate_methods_for_privateuse1_backend,
)

rename_privateuse1_backend("cas_npu")
generate_methods_for_privateuse1_backend()

# 检查设备
print(f"CAS-NPU available: {torch.cas_npu.is_available()}")
print(f"Device count: {torch.cas_npu.device_count()}")

# 创建tensor
device = torch.device("cas_npu:0")
a = torch.randn(3, 3).to(device)
b = torch.randn(3, 3).to(device)

# 执行加法
c = a + b  # 使用CAS-NPU的add.Tensor实现

# 转回CPU
c_cpu = c.cpu()
print(c_cpu)
```

## 扩展功能

要添加新的操作，在`backend/cas_npu_ops.cpp`中：

1. 实现操作函数
2. 使用`TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`注册

例如添加`mul.Tensor`:

```cpp
at::Tensor cas_npu_mul_Tensor(const at::Tensor& self, const at::Tensor& other) {
    // 实现乘法
    ...
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("mul.Tensor", &cas_npu_mul_Tensor);
}
```

## 参考资料

- [PyTorch PrivateUse1文档](https://pytorch.org/docs/stable/notes/extending.html)
- [PyTorch Dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
- [OpenRegistration示例](https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension)


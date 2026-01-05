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
│   ├── cas_npu_module.cpp         # Python绑定
│   └── cas_npu_custom_ops_example.cpp  # 自定义算子示例（量化算子）
├── runtime/                        # Runtime 层
│   ├── cas_npu_runtime.h          # 统一接口定义
│   ├── cmodel/                    # C 模型模拟器
│   │   └── simulator.cpp
│   └── fpga/                      # FPGA 模拟器
│       └── simulator.cpp
├── cas_npu/                       # Python包
│   └── __init__.py                # Python API
├── test/                          # 测试文件
│   ├── test_cas_npu.py            # 基础功能测试
│   ├── test_custom_ops.py         # 自定义算子测试
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
# 基础功能测试
python test/test_cas_npu.py

# 自定义算子测试
python test/test_custom_ops.py
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

### 方式1: 注册PyTorch已有的算子

要添加PyTorch已有的操作（如`mul.Tensor`），在`backend/cas_npu_ops.cpp`中：

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

### 方式2: 注册全新的自定义算子（推荐）

**是的，这套流程完全支持注册PyTorch中不存在的新算子！**

下面以自定义量化算子为例，展示完整的注册流程：

#### 步骤1: 在Runtime层添加底层实现

首先在 `runtime/cas_npu_runtime.h` 中声明函数：

```cpp
// 量化操作: 将float32 tensor量化为int8
CasNpuError casNpuQuantize(
    int8_t* output,
    const float* input,
    size_t num_elements,
    float scale,
    int8_t zero_point);
```

然后在 `runtime/cmodel/simulator.cpp` 或 `runtime/fpga/simulator.cpp` 中实现：

```cpp
CasNpuError casNpuQuantize(
    int8_t* output,
    const float* input,
    size_t num_elements,
    float scale,
    int8_t zero_point) {
    // 量化公式: quantized = round(input / scale) + zero_point
    for (size_t i = 0; i < num_elements; ++i) {
        float quantized_float = input[i] / scale + static_cast<float>(zero_point);
        int quantized_int = static_cast<int>(std::round(quantized_float));
        // Clamp到int8范围 [-128, 127]
        if (quantized_int > 127) quantized_int = 127;
        else if (quantized_int < -128) quantized_int = -128;
        output[i] = static_cast<int8_t>(quantized_int);
    }
    return CAS_NPU_SUCCESS;
}
```

#### 步骤2: 定义算子Schema（在自定义命名空间中）

在 `backend/cas_npu_custom_ops_example.cpp` 中使用 `TORCH_LIBRARY` 定义新命名空间和算子schema：

```cpp
TORCH_LIBRARY(cas_npu, m) {
    // 定义自定义量化算子的schema
    // 参数：输入tensor、量化缩放因子、量化零点
    // 返回：量化后的int8 tensor
    m.def("custom_quantize(Tensor input, float scale, int zero_point) -> Tensor");
}
```

#### 步骤3: 实现算子函数

```cpp
at::Tensor cas_npu_custom_quantize(
    const at::Tensor& input,
    double scale,
    int64_t zero_point) {
    
    // 参数检查
    TORCH_CHECK(input.device().is_privateuseone(), 
                "Expected input tensor on CAS-NPU device");
    TORCH_CHECK(input.scalar_type() == at::kFloat,
                "Input tensor must be float32");
    TORCH_CHECK(scale > 0.0, "Scale must be positive");
    
    // 创建输出tensor（int8类型）
    auto output = at::empty(
        input.sizes(),
        input.options().dtype(at::kInt8)
    );
    
    // 调用runtime执行量化
    auto err = casNpuQuantize(
        output.data_ptr<int8_t>(),
        input.data_ptr<float>(),
        input.numel(),
        static_cast<float>(scale),
        static_cast<int8_t>(zero_point)
    );
    
    TORCH_CHECK(err == CAS_NPU_SUCCESS,
                "CAS-NPU quantize operation failed");
    
    return output;
}
```

#### 步骤4: 为设备注册实现

使用 `TORCH_LIBRARY_IMPL` 为PrivateUse1设备注册实现：

```cpp
TORCH_LIBRARY_IMPL(cas_npu, PrivateUse1, m) {
    m.impl("custom_quantize", &cas_npu_custom_quantize);
}
```

#### 步骤5: 在Python中调用

```python
import torch
import cas_npu

device = torch.device("cas_npu:0")
input_tensor = torch.randn(4, 4, device=device, dtype=torch.float32)

# 调用自定义量化算子
scale = 0.1
zero_point = 0
output = torch.ops.cas_npu.custom_quantize(input_tensor, scale, zero_point)

print(f"输入: {input_tensor}")
print(f"输出: {output}")  # int8类型
```

#### 完整示例和测试

- **完整实现**：`backend/cas_npu_custom_ops_example.cpp`
- **测试脚本**：`test/test_custom_ops.py`

运行测试：
```bash
python test/test_custom_ops.py
```

#### 关键要点

1. **自定义命名空间**：使用 `TORCH_LIBRARY(cas_npu, m)` 创建自定义命名空间，避免与PyTorch核心算子冲突
2. **Schema定义**：使用 `m.def()` 定义算子的输入输出签名
3. **设备实现**：使用 `TORCH_LIBRARY_IMPL(cas_npu, PrivateUse1, m)` 为特定设备注册实现
4. **Python调用**：通过 `torch.ops.cas_npu.算子名` 调用自定义算子
5. **Runtime集成**：在runtime层实现底层计算逻辑，算子层负责tensor管理和参数检查

#### 支持的算子特性

- ✅ 任意输入输出类型（Tensor、Scalar、List等）
- ✅ 可选参数（使用 `Tensor?` 或 `int?` 等）
- ✅ 多返回值（`-> (Tensor, Tensor)`）
- ✅ 复杂参数类型（`int[2]`、`Tensor[]` 等）
- ✅ 与现有流程完全兼容

## 参考资料

- [PyTorch PrivateUse1文档](https://pytorch.org/docs/stable/notes/extending.html)
- [PyTorch Dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
- [OpenRegistration示例](https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension)


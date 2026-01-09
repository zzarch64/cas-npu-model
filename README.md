# CAS-NPU: PyTorch 自定义 NPU 设备扩展

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat&logo=python)
![C++](https://img.shields.io/badge/C++-17-00599c?style=flat&logo=cplusplus)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

**基于 PyTorch PrivateUse1 机制实现的自定义 NPU 设备扩展**

[快速开始](#快速开始) • [添加算子](#添加算子) • [运行网络](#运行网络) • [调试工具](#调试工具) • [开发文档](DEVLOG.md)

</div>

---

## 📖 项目简介

CAS-NPU 是一个使用 PyTorch 的 `PrivateUse1` 机制实现的自定义设备扩展框架。它提供了一套完整的 NPU 后端实现，支持：

- ✅ **完整的设备抽象**：内存管理、设备切换、流同步
- ✅ **渐进式算子开发**：NPU 原生实现 + CPU Fallback 混合模式
- ✅ **多后端支持**：cmodel（调试）/ FPGA / ASIC（生产）
- ✅ **LLM 推理验证**：已通过 Qwen 0.5B 完整前向传播测试

### 当前状态

| 功能 | 状态 | 优先级 | 说明 |
|-----|------|-------|------|
| LeNet Forward | ✅ 完成 | - | CPU vs NPU 输出一致 |
| Qwen 0.5B Forward | ✅ 完成 | - | 完整推理流程 |
| **LoRA Finetune** | 🚧 待开发 | 🔴 P0 | 支持 Qwen 模型训练（最高优先级） |
| CModel 物理内存抽象 | 🚧 待开发 | 🟡 P1 | 从 CPU 虚拟地址迁移到 NPU 物理地址模型 |
| RTL Model (Verilator) | 🚧 待开发 | 🟡 P1 | 基于 Verilator 的 RTL 仿真后端 |
| 编译后端切换 | 🚧 待开发 | 🟡 P1 | CMake/setup.py 编译选项支持多后端 |
| Runtime 架构分层 | 🚧 待开发 | 🟢 P2 | 抽象统一接口，分离 cmodel/rtlmodel/fpga/asic |

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│  Python API Layer                                               │
│  cas_npu/__init__.py - 设备管理、后端注册                         │
├─────────────────────────────────────────────────────────────────┤
│  PyTorch Backend Layer (backend/)                               │
│  ├─ cas_npu_ops.cpp        - 算子实现 (NPU原生 / CPU Fallback)   │
│  ├─ cas_npu_allocator.cpp  - 设备内存分配器                      │
│  ├─ cas_npu_guard.cpp      - DeviceGuard 实现                   │
│  ├─ cas_npu_hooks.cpp      - PrivateUse1 Hooks                  │
│  └─ cas_npu_module.cpp     - Python 绑定 (pybind11)             │
├─────────────────────────────────────────────────────────────────┤
│  Runtime API Layer (runtime/cas_npu_runtime.h)                  │
│  ├─ 内存管理：casNpuMalloc, casNpuFree, casNpuMemcpy            │
│  └─ 计算算子：casNpuMatMul, casNpuAddTensor, ...                │
├─────────────────────────────────────────────────────────────────┤
│  Hardware Implementation Layer                                   │
│  ├─ runtime/cmodel/  - CPU 模拟实现（开发调试）                   │
│  ├─ runtime/fpga/    - FPGA 硬件实现                            │
│  └─ runtime/asic/    - 未来 ASIC 芯片实现                        │
└─────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
npu_cas_extension/
├── backend/                          # PyTorch 后端集成层
│   ├── cas_npu_allocator.h/cpp       # 设备内存分配器
│   ├── cas_npu_guard.h/cpp           # DeviceGuard 实现
│   ├── cas_npu_hooks.h/cpp           # PrivateUse1 Hooks
│   ├── cas_npu_ops.cpp               # 算子实现（核心文件）
│   ├── cas_npu_module.cpp            # Python 绑定
│   └── cas_npu_custom_ops_example.cpp # 自定义算子示例
├── runtime/                          # Runtime 层
│   ├── cas_npu_runtime.h             # Runtime API 定义
│   ├── cas_npu_debug.h               # 调试系统
│   ├── cmodel/simulator.cpp          # C 模型模拟器
│   └── fpga/simulator.cpp            # FPGA 实现
├── cas_npu/                          # Python 包
│   ├── __init__.py                   # 包初始化 & 设备注册
│   └── debug.py                      # Python 调试接口
├── test/                             # 测试文件
│   ├── test_cas_npu.py               # 基础功能测试
│   ├── test_lenet.py                 # LeNet 网络测试
│   ├── test_qwen0.5B.py              # Qwen 模型测试
│   └── test_custom_ops.py            # 自定义算子测试
├── setup.py                          # 构建脚本
├── build_and_test.sh                 # 一键构建测试脚本
├── DEVLOG.md                         # 开发日志（详细设计文档）
└── README.md                         # 本文档
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- C++17 兼容编译器（GCC 7+ / Clang 5+）

### 编译安装

```bash
# 方法1：一键构建并测试
chmod +x build_and_test.sh
./build_and_test.sh

# 方法2：手动构建
python setup.py build_ext --inplace

# 方法3：使用 FPGA 后端构建
CAS_NPU_IMPL=fpga python setup.py build_ext --inplace
```

### 验证安装

```bash
# 运行基础测试
python test/test_cas_npu.py

# 运行网络测试
python test/test_lenet.py
```

### 基础使用

```python
import torch
import cas_npu  # 自动注册后端

# 检查设备
print(f"CAS-NPU available: {torch.cas_npu.is_available()}")
print(f"Device count: {torch.cas_npu.device_count()}")

# 创建设备上的 Tensor
device = torch.device("cas_npu:0")
a = torch.randn(3, 3, device=device)
b = torch.randn(3, 3, device=device)

# 执行计算
c = a + b  # 使用 NPU 原生 add 实现
d = torch.mm(a, b)  # 使用 NPU 原生 mm 实现

# 结果转回 CPU
print(c.cpu())
```

---

## 🔧 添加算子

CAS-NPU 支持两种算子实现方式，可根据开发阶段灵活选择：

### 方式一：NPU 原生实现（高性能）

直接在 NPU 上执行，无 CPU 往返，适用于高频算子。

#### 步骤 1：在 Runtime 层声明 API

在 `runtime/cas_npu_runtime.h` 中添加函数声明：

```cpp
// 例：实现 rsqrt 算子
CasNpuError casNpuRsqrt(
    float* output,
    const float* input,
    size_t num_elements);
```

#### 步骤 2：实现 Runtime 函数

在 `runtime/cmodel/simulator.cpp` 中实现：

```cpp
CasNpuError casNpuRsqrt(
    float* output,
    const float* input,
    size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        output[i] = 1.0f / std::sqrt(input[i]);
    }
    return CAS_NPU_SUCCESS;
}
```

#### 步骤 3：注册 PyTorch 算子

在 `backend/cas_npu_ops.cpp` 中注册：

```cpp
at::Tensor cas_npu_rsqrt(const at::Tensor& self) {
    auto output = at::empty_like(self);
    
    auto err = cas_npu::casNpuRsqrt(
        output.data_ptr<float>(),
        self.data_ptr<float>(),
        self.numel()
    );
    TORCH_CHECK(err == cas_npu::CAS_NPU_SUCCESS, "NPU rsqrt failed");
    
    return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("rsqrt", &cas_npu_rsqrt);
}
```

### 方式二：CPU Fallback（快速开发）

利用 PyTorch 的 CPU 实现，自动处理数据传输。适用于：
- 开发初期快速验证
- 低频算子
- 复杂算子的临时方案

#### 使用统一 cpu_fallback 函数

```cpp
// backend/cas_npu_ops.cpp 中已实现通用 cpu_fallback
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("rsqrt", &cpu_fallback<&at::native::rsqrt>);
    m.impl("pow.Tensor_Scalar", &cpu_fallback<&at::native::pow>);
    // ... 更多算子
}
```

#### 手动实现 Fallback（需要特殊处理时）

```cpp
at::Tensor cas_npu_some_op(const at::Tensor& self) {
    // 1. 拷贝到 CPU
    at::Tensor self_cpu = self.to(at::kCPU);
    
    // 2. 在 CPU 上执行
    at::Tensor result_cpu = at::some_op(self_cpu);
    
    // 3. 拷贝回设备
    return result_cpu.to(self.device());
}
```

### 方式三：自定义命名空间算子

注册 PyTorch 中不存在的全新算子：

```cpp
// 定义 Schema
TORCH_LIBRARY(cas_npu, m) {
    m.def("custom_quantize(Tensor input, float scale, int zero_point) -> Tensor");
}

// 实现算子
at::Tensor cas_npu_custom_quantize(const at::Tensor& input, double scale, int64_t zero_point) {
    // ... 实现
}

// 注册到设备
TORCH_LIBRARY_IMPL(cas_npu, PrivateUse1, m) {
    m.impl("custom_quantize", &cas_npu_custom_quantize);
}
```

Python 调用：

```python
output = torch.ops.cas_npu.custom_quantize(input_tensor, 0.1, 0)
```

---

## 🧠 运行网络

### 基础：移动模型到设备

```python
import torch
import torch.nn as nn
import cas_npu

device = torch.device('cas_npu:0')

# 方法1：创建后移动
model = MyModel()
model = model.to(device)

# 方法2：直接在设备上创建
with torch.device(device):
    model = MyModel()

# 准备输入
input_data = torch.randn(batch_size, ...).to(device)

# 推理
with torch.no_grad():
    output = model(input_data)
```

### 示例：运行 LeNet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import cas_npu

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 运行推理
device = torch.device('cas_npu:0')
model = LeNet().to(device)
x = torch.randn(4, 1, 28, 28).to(device)

with torch.no_grad():
    output = model(x)
    print(output.cpu())
```

### 示例：运行 Qwen 0.5B

```python
import torch
import cas_npu
from transformers import AutoModel, AutoTokenizer

device = torch.device('cas_npu:0')

# 加载模型
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype=torch.float32,
)

# 移动到 NPU
model = model.to(device)
model.eval()

# 推理
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

with torch.no_grad():
    outputs = model(input_ids)
    hidden_states = outputs.last_hidden_state
    print(f"Output shape: {hidden_states.shape}")
```

---

## 🐛 调试工具

### 环境变量控制

```bash
# 启用调试打印
CAS_NPU_DEBUG=1 python your_script.py

# 设置详细程度 (1-3)
CAS_NPU_DEBUG_LEVEL=2 python your_script.py
```

| Level | 显示内容 |
|-------|---------|
| 1 | 仅算子执行信息 |
| 2 | 算子执行 + 数据传输（默认） |
| 3 | 全部信息（含 Runtime 层） |

### Python API 控制

```python
import cas_npu.debug as debug

# 启用/禁用
debug.enable(level=2)
debug.disable()

# 临时调试模式
with debug.debug_mode(level=3):
    output = model(input)
```

### 输出格式说明

```
[NPU]      绿色 - NPU 原生实现（高性能）
[CPU←→NPU] 黄色 - 显式 CPU Fallback
[VIEW]     青色 - View 操作（仅修改 metadata）
[CPU]      红色 - 纯 CPU Fallback
[COPY]     蓝色 - 数据拷贝操作

数据传输:
[H→D] - Host 到 Device
[D→H] - Device 到 Host
[D→D] - Device 到 Device
```

### 示例输出

```
[NPU] mm [128x768] @ [768x3072]
[CPU←→NPU] rsqrt [98304]
    ↳ [D→H] 384.00 KB
    ↳ [H→D] 384.00 KB
[VIEW] reshape
```

---

## 📊 算子支持状态

### NPU 原生实现（高性能）

| 算子 | Runtime API | 用途 |
|-----|-------------|------|
| `mm` | `casNpuMatMul` | Linear 层、投影 |
| `bmm` | `casNpuBatchMatMul` | Attention 计算 |
| `add.Tensor` | `casNpuAddTensor` | 残差连接 |

### CPU Fallback（待优化）

| 类别 | 算子 | 优先级 |
|-----|------|-------|
| RMSNorm | `rsqrt`, `pow`, `mean.dim` | 🔴 高 |
| 激活函数 | `silu`, `relu` | 🔴 高 |
| Rotary Embedding | `cos`, `sin` | 🔴 高 |
| 基础运算 | `mul.Tensor`, `div.Tensor` | 🟡 中 |
| Attention | `softmax`, `scaled_dot_product_attention` | 🟡 中 |

### View 操作（零开销）

`view`, `reshape`, `transpose`, `permute`, `unsqueeze`, `squeeze`, `expand`, `slice`, `select`, `as_strided`, `t`, `detach`

> 详细开发计划请参考 [DEVLOG.md](DEVLOG.md)

---

## 🗺️ 开发路线图

### 🚧 待开发功能（按优先级排序）

#### 1. 🔴 LoRA Finetune 支持（最高优先级）

**问题**：当前仅支持推理（前向传播），不支持训练（反向传播）。

**目标**：在 CAS-NPU 上实现 Qwen 0.5B 的 LoRA 微调，验证训练支持。

**需要实现的功能**：

| 类别 | 需求 | 优先级 |
|-----|------|-------|
| Autograd 支持 | 实现 `backward()` 相关算子 | 🔴 P0 |
| 梯度计算 | `mm` 反向、`add` 反向等 | 🔴 P0 |
| 优化器支持 | AdamW 等优化器在设备上执行 | 🟡 P1 |
| LoRA 层 | 低秩适配器的高效实现 | 🟡 P1 |
| 混合精度 | FP16/BF16 训练支持 | 🟢 P2 |

**实现路径**：
1. 实现基础反向传播算子（从 Fallback 开始）
2. 验证简单网络（如 LeNet）的训练
3. 实现 LoRA 相关算子的 NPU 原生版本
4. 完成 Qwen LoRA 微调端到端流程

#### 2. CModel 物理内存抽象

**问题**：当前 CModel 直接使用 CPU 虚拟地址（`malloc`/`free`），无法真实模拟 NPU 的物理内存访问行为。

**目标**：
- 维护一套独立的 NPU 物理地址空间
- CModel 通过物理地址进行访存模拟
- 为后续 RTL Model 和硬件对接打好基础

```
当前实现（问题）：
┌─────────────────────────────────────────────────────────────┐
│  casNpuMalloc() ──▶ CPU malloc() ──▶ 返回 CPU 虚拟地址       │
│  casNpuMemcpy() ──▶ CPU memcpy() ──▶ 直接操作 CPU 内存       │
└─────────────────────────────────────────────────────────────┘

目标实现：
┌─────────────────────────────────────────────────────────────┐
│  NPU Physical Address Space (模拟)                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  0x0000_0000 ─┬─ Weight Memory Region               │   │
│  │               ├─ Activation Memory Region           │   │
│  │               └─ ... (可配置布局)                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  casNpuMalloc() ──▶ 分配物理地址 ──▶ 返回 NPU 物理地址       │
│  casNpuMemcpy() ──▶ 物理地址转换 ──▶ 操作模拟 RAM            │
└─────────────────────────────────────────────────────────────┘
```

#### 3. RTL Model 支持 (Verilator)

**目标**：基于 Verilator 将 NPU IP 的 RTL 代码封装为仿真后端。

**架构设计**：

```
┌─────────────────────────────────────────────────────────────────┐
│  Runtime API Layer                                              │
│  casNpuMatMul(), casNpuAddTensor(), ...                        │
├─────────────────────────────────────────────────────────────────┤
│  RTL Model Backend (runtime/rtlmodel/)                          │
│  ├─ verilator_wrapper.cpp    - Verilator 仿真控制               │
│  ├─ axi_driver.cpp           - AXI 总线驱动                     │
│  ├─ command_packet.h         - 数据/命令包定义                   │
│  └─ ram_interface.cpp        - RAM 模型接口                     │
├─────────────────────────────────────────────────────────────────┤
│  Verilator Generated Code                                       │
│  ├─ Vnpu_top.h               - NPU 顶层模块                     │
│  └─ Vnpu_top__ALL.a          - 编译后的仿真库                   │
├─────────────────────────────────────────────────────────────────┤
│  NPU RTL Design                                                 │
│  ├─ npu_top.v                - 顶层模块 (AXI Slave)             │
│  ├─ matrix_engine.v          - 矩阵计算单元                     │
│  └─ ...                                                         │
└─────────────────────────────────────────────────────────────────┘
```

**AXI 接口与命令包设计**：

```cpp
// 命令包格式（示例）
struct NpuCommandPacket {
    uint32_t opcode;        // 操作码: MATMUL, ADD, MEMCPY, ...
    uint32_t src1_addr;     // 源地址1 (物理地址)
    uint32_t src2_addr;     // 源地址2 (物理地址)
    uint32_t dst_addr;      // 目标地址 (物理地址)
    uint32_t param[4];      // 参数: M, K, N, alpha, ...
};

// AXI 驱动接口
class AxiDriver {
    void writeCommand(const NpuCommandPacket& cmd);
    void waitComplete();
    void readStatus(uint32_t* status);
};
```

#### 4. 编译后端切换支持

**目标**：通过编译选项支持 CModel、RTLModel、FPGA、ASIC 后端的切换。

**编译命令**：

```bash
# CModel 后端（默认，快速开发调试）
python setup.py build_ext --inplace
# 或
CAS_NPU_BACKEND=cmodel python setup.py build_ext --inplace

# RTL Model 后端（RTL 仿真验证）
CAS_NPU_BACKEND=rtlmodel python setup.py build_ext --inplace

# FPGA 后端（硬件验证）
CAS_NPU_BACKEND=fpga python setup.py build_ext --inplace

# ASIC 后端（芯片驱动）
CAS_NPU_BACKEND=asic python setup.py build_ext --inplace
```

**setup.py 改进**：

```python
# 读取后端选择
backend = os.environ.get('CAS_NPU_BACKEND', 'cmodel')

# 根据后端选择源文件
backend_sources = {
    'cmodel':   ['runtime/cmodel/backend.cpp'],
    'rtlmodel': ['runtime/rtlmodel/backend.cpp', 
                 'runtime/rtlmodel/verilator_wrapper.cpp'],
    'fpga':     ['runtime/fpga/backend.cpp'],
    'asic':     ['runtime/asic/backend.cpp'],
}
```

#### 5. Runtime 架构重构

**问题**：当前 Runtime 层的抽象不够清晰，`cas_npu_runtime.h` 中的 API 声明与具体实现耦合过紧。

**目标架构**：

```
┌─────────────────────────────────────────────────────────────────┐
│  Runtime API Layer (runtime/cas_npu_runtime.h)                  │
│  ├─ 统一接口定义（纯虚函数 / 函数指针表）                          │
│  └─ 后端无关的通用逻辑                                           │
├─────────────────────────────────────────────────────────────────┤
│  Backend Abstraction Layer (runtime/backend_interface.h)        │
│  ├─ CasNpuBackend 抽象基类                                      │
│  └─ 运行时后端选择 & 动态加载                                     │
├─────────────────────────────────────────────────────────────────┤
│  Concrete Implementations                                        │
│  ├─ runtime/cmodel/backend.cpp    - CPU 模拟（物理内存模型）      │
│  ├─ runtime/rtlmodel/backend.cpp  - Verilator RTL 仿真          │
│  ├─ runtime/fpga/backend.cpp      - FPGA 硬件驱动               │
│  └─ runtime/asic/backend.cpp      - ASIC 芯片驱动               │
└─────────────────────────────────────────────────────────────────┘
```

**预期收益**：
- 清晰的接口抽象，便于添加新后端
- 支持运行时动态切换后端（不需要重新编译）
- 更好的代码复用和测试隔离

---

## 🧪 测试

```bash
# 基础功能测试
python test/test_cas_npu.py

# LeNet 网络测试
python test/test_lenet.py

# Qwen 模型测试（需要 transformers）
python test/test_qwen0.5B.py

# 自定义算子测试
python test/test_custom_ops.py

# 带调试输出测试
CAS_NPU_DEBUG_LEVEL=2 python test/test_lenet.py
```

---

## 📚 参考资料

- [PyTorch PrivateUse1 文档](https://pytorch.org/docs/stable/notes/extending.html)
- [PyTorch Dispatcher 详解](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
- [OpenRegistration 官方示例](https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension)

---

## 📝 License

MIT License

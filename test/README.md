# CAS-NPU 测试套件

本目录包含 CAS-NPU 扩展的完整测试套件，用于验证自定义设备的功能和正确性。

## 📁 目录结构

```
test/
├── run_all_tests.py          # 运行所有测试的脚本
├── test_framework.py          # 测试框架和工具函数
├── unit/                      # 单元测试
│   ├── test_basic_ops.py     # 基础操作测试（add_, copy_）
│   ├── test_gradient.py       # 梯度计算测试
│   ├── test_addmm.py         # addmm 操作测试
│   ├── test_linear.py         # Linear 层测试
│   ├── operators/            # 算子精度测试
│   │   ├── test_operator_accuracy.py    # 算子精度测试
│   │   └── test_addmm_detailed.py      # 详细 addmm 测试
│   └── memory/                # 内存和数据传输测试
│       └── test_copy_from_detailed.py   # 详细拷贝测试
├── integration/               # 集成测试
│   ├── test_cas_npu.py       # 基础功能测试
│   ├── test_concept.py       # 概念验证测试
│   ├── test_custom_ops.py    # 自定义算子测试
│   ├── model/                 # 模型层测试
│   │   ├── test_layer_by_layer.py       # 逐层测试
│   │   ├── test_ffn_layer.py            # FFN 层测试
│   │   ├── test_ffn_step_by_step.py     # FFN 逐步测试
│   │   └── test_cpu_vs_npu.py           # CPU vs NPU 对比
│   └── attention/             # Attention 测试
│       ├── test_attention_computation.py # Attention 计算测试
│       ├── test_attention_mask.py       # Attention mask 测试
│       └── test_attention_mask_detailed.py # 详细 attention mask 测试
└── tools/                     # 测试工具
    ├── gradient_analyzer.py   # 梯度 NaN 分析工具
    ├── test_nan_diagnosis.py # NaN 诊断工具
    └── test_asan.py           # AddressSanitizer 测试
```

## 🚀 快速开始

### 1. 编译扩展

在运行测试之前，需要先编译 C++ 扩展：

```bash
# 在项目根目录
python setup.py build_ext --inplace
```

### 2. 运行测试

#### 运行所有测试（推荐）

使用 `run_all_tests.py` 脚本可以一次性运行所有测试：

```bash
# 运行所有测试
python test/run_all_tests.py

# 只运行单元测试
python test/run_all_tests.py --unit

# 只运行集成测试
python test/run_all_tests.py --integration

# 详细输出
python test/run_all_tests.py -vv

# 安静模式（只显示结果）
python test/run_all_tests.py -q

# 包含测试工具
python test/run_all_tests.py --tools
```

#### 运行单个测试文件

```bash
# 基础操作测试
python test/unit/test_basic_ops.py

# 梯度计算测试
python test/unit/test_gradient.py

# addmm 操作测试
python test/unit/test_addmm.py

# Linear 层测试
python test/unit/test_linear.py

# 算子精度测试
python test/unit/operators/test_operator_accuracy.py
python test/unit/operators/test_addmm_detailed.py

# 内存拷贝测试
python test/unit/memory/test_copy_from_detailed.py
```

#### 运行集成测试

```bash
# 基础功能测试
python test/integration/test_cas_npu.py

# 概念验证测试（无需编译）
python test/integration/test_concept.py

# 自定义算子测试
python test/integration/test_custom_ops.py

# 模型层测试
python test/integration/model/test_layer_by_layer.py
python test/integration/model/test_ffn_layer.py
python test/integration/model/test_ffn_step_by_step.py
python test/integration/model/test_cpu_vs_npu.py

# Attention 测试
python test/integration/attention/test_attention_computation.py
python test/integration/attention/test_attention_mask.py
python test/integration/attention/test_attention_mask_detailed.py
```

#### 使用测试工具

```bash
# 梯度 NaN 分析
python test/tools/gradient_analyzer.py

# NaN 诊断
python test/tools/test_nan_diagnosis.py

# AddressSanitizer 测试
python test/tools/test_asan.py
```

### 3. 测试参数

所有测试都支持统一的命令行参数：

```bash
# 详细输出（-v: normal, -vv: verbose, -vvv: debug）
python test/unit/test_basic_ops.py -vv

# 安静模式（只显示结果）
python test/unit/test_basic_ops.py -q

# 指定设备
python test/unit/test_basic_ops.py --device cas_npu:0

# 指定容差
python test/unit/test_basic_ops.py --tolerance 1e-6
```

## 📋 测试文件详细说明

### 单元测试 (test/unit/)

#### `test_basic_ops.py` - 基础操作测试

**测试内容**:
- `add_.Tensor` 操作（原地加法）
- 梯度累积模拟（使用 `add_`）
- 大 tensor 拷贝（CPU <-> Device）
- 部分拷贝（包含 NaN 的情况）

**运行方式**:
```bash
python test/unit/test_basic_ops.py [-v] [-q] [--device DEVICE] [--tolerance TOL]
```

---

#### `test_gradient.py` - 梯度计算测试

**测试内容**:
- 梯度 tensor 创建过程
- 梯度流动过程（前向和反向传播）
- 梯度数值验证
- 手动梯度计算验证

**运行方式**:
```bash
python test/unit/test_gradient.py [-v] [-q] [--device DEVICE] [--tolerance TOL]
```

---

#### `test_addmm.py` - addmm 操作测试

**测试内容**:
- addmm 前向传播
- addmm 梯度计算
- 梯度数值验证（与手动计算对比）
- 逐步检查梯度计算过程

**运行方式**:
```bash
python test/unit/test_addmm.py [-v] [-q] [--device DEVICE] [--tolerance TOL]
```

---

#### `test_linear.py` - Linear 层测试

**测试内容**:
- Linear 层前向传播
- 手动矩阵乘法验证
- 添加偏置验证
- Linear 层反向传播
- 梯度验证

**运行方式**:
```bash
python test/unit/test_linear.py [-v] [-q] [--device DEVICE] [--tolerance TOL]
```

---

#### `operators/test_operator_accuracy.py` - 算子精度测试

**测试内容**:
- 基础算子测试 (mm, bmm, add, addmm)
- 模型第一层输出对比
- 逐步检查每个 transformer layer

**运行方式**:
```bash
python test/unit/operators/test_operator_accuracy.py [-v] [-q] [--device DEVICE] [--tolerance TOL] [--model-path PATH] [--num-layers N]
```

---

#### `operators/test_addmm_detailed.py` - 详细 addmm 测试

**测试内容**:
- 基本 addmm 操作
- 使用实际模型权重测试 (gate_proj, up_proj, down_proj)

**运行方式**:
```bash
python test/unit/operators/test_addmm_detailed.py [-v] [-q] [--device DEVICE] [--tolerance TOL] [--model-path PATH]
```

---

#### `memory/test_copy_from_detailed.py` - 详细拷贝测试

**测试内容**:
- 基本拷贝测试 (CPU->NPU, NPU->CPU, NPU->NPU)
- 非 contiguous tensor 拷贝 (transpose, slice, view)
- 3D tensor 拷贝
- 模型数据传递测试

**运行方式**:
```bash
python test/unit/memory/test_copy_from_detailed.py [-v] [-q] [--device DEVICE] [--tolerance TOL] [--model-path PATH]
```

---

### 集成测试 (test/integration/)

#### `model/test_layer_by_layer.py` - 逐层测试

**测试内容**:
- Embedding 层对比
- 逐层检查 transformer layers
- 最终输出对比

**运行方式**:
```bash
python test/integration/model/test_layer_by_layer.py [-v] [-q] [--device DEVICE] [--tolerance TOL] [--model-path PATH] [--num-layers N]
```

---

#### `model/test_ffn_layer.py` - FFN 层测试

**测试内容**:
- Attention 输出对比
- FFN 输出对比
- Layer 输出对比
- FFN 关键操作测试 (linear, SiLU)

**运行方式**:
```bash
python test/integration/model/test_ffn_layer.py [-v] [-q] [--device DEVICE] [--tolerance TOL] [--model-path PATH]
```

---

#### `model/test_ffn_step_by_step.py` - FFN 逐步测试

**测试内容**:
- Input layer norm
- Gate projection
- Up projection
- SiLU activation
- Multiply (SiLU(gate) * up)
- Down projection
- Complete FFN output

**运行方式**:
```bash
python test/integration/model/test_ffn_step_by_step.py [-v] [-q] [--device DEVICE] [--tolerance TOL] [--model-path PATH]
```

---

#### `model/test_cpu_vs_npu.py` - CPU vs NPU 对比

**测试内容**:
- Forward pass 对比
- Generation 对比

**运行方式**:
```bash
python test/integration/model/test_cpu_vs_npu.py [-v] [-q] [--device DEVICE] [--tolerance TOL] [--model-path PATH] [--max-new-tokens N]
```

---

#### `attention/test_attention_computation.py` - Attention 计算测试

**测试内容**:
- Attention 输入输出对比
- Q @ K^T (bmm) 测试
- Softmax 测试
- Attention @ V (bmm) 测试

**运行方式**:
```bash
python test/integration/attention/test_attention_computation.py [-v] [-q] [--device DEVICE] [--tolerance TOL] [--model-path PATH]
```

---

#### `attention/test_attention_mask.py` - Attention mask 测试

**测试内容**:
- Forward pass 中 attention_mask 的使用
- Generation 中 attention_mask 的使用
- masked_fill_ 操作测试

**运行方式**:
```bash
python test/integration/attention/test_attention_mask.py [-v] [-q] [--device DEVICE] [--tolerance TOL] [--model-path PATH]
```

---

#### `attention/test_attention_mask_detailed.py` - 详细 attention mask 测试

**测试内容**:
- attention_mask 对输出的影响
- Hook masked_fill_ 调用

**运行方式**:
```bash
python test/integration/attention/test_attention_mask_detailed.py [-v] [-q] [--device DEVICE] [--tolerance TOL] [--model-path PATH]
```

---

#### `test_cas_npu.py` - 基础功能测试

**用途**: 测试 CAS-NPU 扩展的基础功能

**测试内容**:
1. 设备可用性检查
2. Tensor 创建和设备转移
3. add.Tensor 操作
4. 设备切换
5. Tensor 方法

**运行方式**:
```bash
python test/integration/test_cas_npu.py
```

**前置条件**: 需要先编译 C++ 扩展

---

#### `test_concept.py` - 概念验证测试

**用途**: 纯 Python 实现的概念验证，无需编译 C++ 扩展

**特点**:
- 使用 NumPy 模拟 CAS-NPU 设备操作
- 验证 PrivateUse1 机制的设计正确性
- 手动注册操作实现

**运行方式**:
```bash
python test/integration/test_concept.py
```

**适用场景**: 
- 在编译 C++ 扩展之前验证设计思路
- 快速验证 PrivateUse1 机制是否正常工作

---

#### `test_custom_ops.py` - 自定义算子测试

**用途**: 测试自定义量化算子示例

**运行方式**:
```bash
python test/integration/test_custom_ops.py
```

---

### 测试工具 (test/tools/)

#### `gradient_analyzer.py` - 梯度 NaN 分析工具

**用途**: 分析梯度 tensor 中 NaN 的分布模式，帮助诊断梯度计算问题

**功能**:
- NaN 分布分析（按行、按列）
- NaN 聚类分析
- NaN 位置分析
- 期望梯度对比
- 内存布局分析

**运行方式**:
```bash
python test/tools/gradient_analyzer.py [-v] [-q] [--device DEVICE]
```

---

#### `test_nan_diagnosis.py` - NaN 诊断工具

**用途**: 检查推理和训练过程中 NaN 的来源

**运行方式**:
```bash
python test/tools/test_nan_diagnosis.py
```

---

#### `test_asan.py` - AddressSanitizer 测试

**用途**: 测试 masked_fill_ 相关的操作，避免加载完整模型

**测试内容**:
- 简单的 masked_fill_
- Attention mask 处理
- 多次调用 masked_fill_
- 不同大小的 tensor

**运行方式**:
```bash
python test/tools/test_asan.py [-v] [-q] [--device DEVICE] [--tolerance TOL]
```

---

## 🛠️ 测试框架

所有单元测试都使用统一的测试框架 (`test_framework.py`)，提供：

- **统一的 tensor 检查函数**: `check_tensor()`
- **梯度验证函数**: `verify_tensor_match()`
- **NaN 分析函数**: `analyze_nan_distribution()`
- **可配置的详细程度**: QUIET, NORMAL, VERBOSE, DEBUG
- **统一的命令行参数**: `-v`, `-q`, `--device`, `--tolerance`

## 📝 测试依赖

### 必需依赖
- PyTorch (>= 1.13.0)
- NumPy

### 可选依赖
- `transformers` - 用于某些模型测试
- `peft` - 用于 LoRA 测试

## 🔍 测试覆盖范围

| 测试文件 | 设备注册 | 基础操作 | 梯度计算 | 神经网络 | 大模型 |
|---------|---------|---------|---------|---------|--------|
| `test_cas_npu.py` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `test_concept.py` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `test_basic_ops.py` | ✅ | ✅ | ✅ | ❌ | ❌ |
| `test_gradient.py` | ✅ | ✅ | ✅ | ❌ | ❌ |
| `test_addmm.py` | ✅ | ✅ | ✅ | ❌ | ❌ |
| `test_linear.py` | ✅ | ✅ | ✅ | ✅ | ❌ |

## 💡 开发建议

1. **开发新功能时**: 先运行 `test/integration/test_concept.py` 验证设计，再实现 C++ 版本
2. **添加新操作时**: 在 `test/unit/` 中添加对应的单元测试
3. **测试复杂模型时**: 参考 `examples/` 目录下的示例代码
4. **调试梯度问题**: 使用 `test/tools/gradient_analyzer.py` 分析 NaN 分布

## 🔗 相关文档

- [主 README](../README.md) - 项目总体介绍
- [示例代码](../examples/README.md) - 使用示例
- [开发日志](../DEVLOG.md) - 开发过程记录

## 📄 许可证

与主项目保持一致。

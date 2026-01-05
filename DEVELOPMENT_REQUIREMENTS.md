# CAS NPU Extension 开发需求手册

## 项目概述

本项目旨在为 PyTorch 提供 CAS NPU 后端支持，实现核心算子以支持深度学习模型（特别是 LLM）在 NPU 上的运行。

---

## 开发思路

### 核心策略：先跑通，再优化

采用**渐进式开发**策略，分为两个主要阶段：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段一：功能验证（Fallback 模式）                                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                        │
│  目标：让模型完整跑起来，验证框架正确性                                    │
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ PyTorch  │───▶│ Device→  │───▶│ CPU 计算 │───▶│ →Device  │          │
│  │ 算子调用  │    │ CPU 拷贝 │    │ (现成实现)│    │ 拷贝回   │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│                                                                         │
│  ✓ 快速实现所有算子    ✓ 验证模型正确性    ✓ 建立测试基准               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段二：性能优化（NPU 原生实现）                                         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                        │
│  目标：逐个替换为 NPU 原生算子，提升性能                                   │
│                                                                         │
│  ┌──────────┐    ┌─────────────────────────────────┐                   │
│  │ PyTorch  │───▶│ NPU Runtime API 直接执行        │                   │
│  │ 算子调用  │    │ (无 CPU 往返，最优性能)          │                   │
│  └──────────┘    └─────────────────────────────────┘                   │
│                                                                         │
│  ✓ 消除内存拷贝    ✓ 充分利用 NPU 算力    ✓ 达到生产性能                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 为什么采用这种策略？

| 优势 | 说明 |
|-----|------|
| **降低开发风险** | 先用成熟的 CPU 实现验证整体框架，避免同时调试框架和算子 |
| **快速迭代** | Fallback 模式下可以快速支持新算子，让模型跑起来 |
| **明确优化方向** | 通过 profiling 确定性能瓶颈，按优先级优化最关键的算子 |
| **保持可测试性** | Fallback 实现作为参考，可用于验证 NPU 原生实现的正确性 |

### 具体迭代步骤

```
1. 新算子需求 ──▶ 2. 实现 Fallback 版本 ──▶ 3. 模型测试通过
                         │
                         ▼
4. 性能分析 ◀────────────┘
      │
      ▼
5. 高频算子优先实现 NPU 原生版本
      │
      ▼
6. 对比 Fallback 验证正确性 ──▶ 7. 替换为原生实现
```

### 当前状态

- ✅ **阶段一基本完成**：Qwen 0.5B 模型已能完整运行
- ⏳ **阶段二进行中**：已实现 `mm`、`bmm`、`add.Tensor` 的 NPU 原生版本
- 📋 **下一步**：按优先级实现 RMSNorm 相关算子（rsqrt, mul, pow, mean）

---

## 架构设计

### 内存模型：显式 Copy 模式

采用类似 CUDA GPU 的独立内存架构，设备内存和 CPU 内存不共享，需要显式进行数据拷贝。

```
┌─────────────────────────────────────────────────────────────┐
│                      Host (CPU) Memory                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
         casNpuMemcpy(HOST_TO_DEVICE) ↓ ↑ casNpuMemcpy(DEVICE_TO_HOST)
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                    Device (NPU) Memory                       │
│                                                              │
│   casNpuMalloc() 分配    casNpuFree() 释放                   │
└─────────────────────────────────────────────────────────────┘
```

### 内存拷贝方向（类似 cudaMemcpyKind）

```cpp
enum CasNpuMemcpyKind {
    CAS_NPU_MEMCPY_HOST_TO_HOST = 0,      // CPU -> CPU
    CAS_NPU_MEMCPY_HOST_TO_DEVICE = 1,    // CPU -> Device
    CAS_NPU_MEMCPY_DEVICE_TO_HOST = 2,    // Device -> CPU
    CAS_NPU_MEMCPY_DEVICE_TO_DEVICE = 3,  // Device -> Device
    CAS_NPU_MEMCPY_DEFAULT = 4            // 自动检测
};
```

### 算子实现策略

| 实现方式 | 描述 | 性能 | 适用场景 |
|---------|------|------|---------|
| **NPU 原生实现** | 直接在 NPU 上执行，无内存拷贝 | ⭐⭐⭐ 最优 | 高频算子（mm, bmm, add） |
| **CPU Fallback (显式 Copy)** | Device→CPU→计算→Device | ⭐ 较慢 | 开发阶段临时方案 |
| **View 操作** | 仅修改 metadata，无数据拷贝 | ⭐⭐⭐ 最优 | reshape, transpose, slice 等 |

### 代码架构

```
┌─────────────────────────────────────────────────────────────┐
│  PyTorch 算子层 (backend/cas_npu_ops.cpp)                    │
│  - NPU 原生实现：直接调用 Runtime API                         │
│  - CPU Fallback：Device→CPU→计算→Device                      │
│  - View 操作：仅修改 tensor metadata                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  NPU Runtime API (runtime/cas_npu_runtime.h)                │
│  - 内存管理：casNpuMalloc, casNpuFree, casNpuMemcpy         │
│  - 计算算子：casNpuMatMul, casNpuAddTensor, ...             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  硬件实现层                                                  │
│  - runtime/cmodel/   : CPU 模拟实现（开发调试用）            │
│  - runtime/fpga/     : FPGA 硬件实现                        │
│  - runtime/asic/     : 未来 ASIC 芯片实现                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Qwen 0.5B 推理算子进度表

### 📊 总体进度

| 类别 | NPU原生实现 | CPU Fallback | View操作 | 总计 |
|-----|------------|--------------|---------|------|
| 数量 | 3 | 17 | 13 | 33 |
| 状态 | ✅ 已完成 | ⚠️ 临时方案 | ✅ 已完成 | - |

### ✅ NPU 原生实现（高性能）

这些算子直接在 NPU 上执行，无需 CPU 往返，是性能关键路径。

| 算子 | Runtime API | 用途 | 状态 |
|-----|-------------|------|------|
| `mm` | `casNpuMatMul` | Linear 层、投影层 | ✅ 已实现 |
| `bmm` | `casNpuBatchMatMul` | Attention (Q@K^T, scores@V) | ✅ 已实现 |
| `add.Tensor` | `casNpuAddTensor` | 残差连接、偏置加法 | ✅ 已实现 |

### ⚠️ CPU Fallback 实现（需优化为 NPU 原生）

这些算子当前使用 Device→CPU→计算→Device 模式，每次调用都有内存拷贝开销。

#### RMSNorm 相关（高优先级）
| 算子 | 待实现 Runtime API | 用途 | 调用频率 |
|-----|-------------------|------|---------|
| `rsqrt` | `casNpuRsqrt` | 1/sqrt(x)，RMSNorm 核心 | 每层 2 次 |
| `pow.Tensor_Scalar` | `casNpuPow` | x^2，计算方差 | 每层 2 次 |
| `mean.dim` | `casNpuMean` | 维度均值 | 每层 2 次 |

#### Rotary Embedding 相关（高优先级）
| 算子 | 待实现 Runtime API | 用途 | 调用频率 |
|-----|-------------------|------|---------|
| `cos` | `casNpuCos` | 位置编码 | 每层 1 次 |
| `sin` | `casNpuSin` | 位置编码 | 每层 1 次 |

#### 基础数学运算（中优先级）
| 算子 | 待实现 Runtime API | 用途 | 调用频率 |
|-----|-------------------|------|---------|
| `mul.Tensor` | `casNpuMulTensor` | 逐元素乘法 | 高 |
| `mul.Scalar` | `casNpuMulScalar` | 标量乘法 | 高 |
| `add.Scalar` | `casNpuAddScalar` | 标量加法 | 中 |
| `sub.Tensor` | `casNpuSubTensor` | 减法 | 低 |
| `div.Tensor` | `casNpuDivTensor` | 除法 | 低 |
| `neg` | `casNpuNeg` | 取负 | 低 |
| `sqrt` | `casNpuSqrt` | 平方根 | 低 |

#### 激活函数（高优先级）
| 算子 | 待实现 Runtime API | 用途 | 调用频率 |
|-----|-------------------|------|---------|
| `silu` | `casNpuSiLU` | SiLU 激活 (FFN) | 每层 1 次 |

#### Attention 相关（高优先级）
| 算子 | 待实现 Runtime API | 用途 | 调用频率 |
|-----|-------------------|------|---------|
| `softmax.int` | `casNpuSoftmax` | Attention 归一化 | 每层 1 次 |
| `scaled_dot_product_attention` | `casNpuSDPA` | 融合 Attention | 每层 1 次 |

#### 其他（低优先级）
| 算子 | 待实现 Runtime API | 用途 | 调用频率 |
|-----|-------------------|------|---------|
| `embedding` | `casNpuEmbedding` | Token 嵌入 | 仅输入层 |
| `cat` | `casNpuCat` | KV Cache 拼接 | 每层 2 次 |
| `clone` | `casNpuClone` | 张量复制 | 低 |
| `contiguous` | `casNpuContiguous` | 内存连续化 | 低 |

### ✅ View 操作（无需优化）

这些操作仅修改 tensor 的 metadata（shape, stride, offset），不涉及数据拷贝，已是最优实现。

| 算子 | 用途 | 状态 |
|-----|------|------|
| `view` | 改变形状 | ✅ 原生实现 |
| `reshape` | 改变形状 | ✅ 原生实现 |
| `transpose.int` | 交换维度 | ✅ 原生实现 |
| `permute` | 重排维度 | ✅ 原生实现 |
| `unsqueeze` | 插入维度 | ✅ 原生实现 |
| `squeeze` / `squeeze.dim` | 移除维度 | ✅ 原生实现 |
| `expand` | 广播扩展 | ✅ 原生实现 |
| `slice.Tensor` | 切片 | ✅ 原生实现 |
| `select.int` | 选择索引 | ✅ 原生实现 |
| `as_strided` | 自定义 stride | ✅ 原生实现 |
| `t` | 2D 转置 | ✅ 原生实现 |
| `detach` | 分离梯度 | ✅ 原生实现 |
| `_reshape_alias` | reshape 别名 | ✅ 原生实现 |

---

## 开发优先级

### 🔴 P0: 性能关键（每层多次调用）

**目标：消除最高频的 CPU 往返**

| 算子 | 预期性能提升 | 复杂度 |
|-----|-------------|-------|
| `rsqrt` | 高 | 低 |
| `mul.Tensor` / `mul.Scalar` | 高 | 低 |
| `silu` | 高 | 低 |
| `cos` / `sin` | 中 | 低 |
| `pow.Tensor_Scalar` | 中 | 低 |
| `mean.dim` | 中 | 中 |

### 🟡 P1: 重要优化

| 算子 | 预期性能提升 | 复杂度 |
|-----|-------------|-------|
| `softmax` | 高 | 中 |
| `scaled_dot_product_attention` | 很高（融合算子） | 高 |

### 🟢 P2: 完整性

| 算子 | 说明 |
|-----|------|
| `embedding` | 仅输入层调用 |
| `cat` | KV Cache 操作 |
| 其他基础算子 | 按需实现 |

---

## 开发计划

### Phase 1: 核心矩阵运算 ✅ 已完成
- [x] MM (矩阵乘法) - `casNpuMatMul`
- [x] BMM (批量矩阵乘法) - `casNpuBatchMatMul`
- [x] Add (张量加法) - `casNpuAddTensor`
- [x] 显式 Copy 内存模型 - `casNpuMemcpy` with direction

### Phase 2: RMSNorm 原生实现 ⏳ 进行中
- [ ] `casNpuRsqrt` - rsqrt
- [ ] `casNpuPow` - pow
- [ ] `casNpuMean` - mean.dim
- [ ] `casNpuMulTensor` / `casNpuMulScalar` - mul

### Phase 3: 激活函数 & 位置编码
- [ ] `casNpuSiLU` - silu 激活
- [ ] `casNpuCos` / `casNpuSin` - Rotary Embedding

### Phase 4: Attention 优化
- [ ] `casNpuSoftmax` - softmax
- [ ] `casNpuSDPA` - Scaled Dot-Product Attention（融合算子）

### Phase 5: 其他算子 & 性能优化
- [ ] Embedding
- [ ] 矩阵乘法性能优化（BLAS/专用库）
- [ ] Caching Allocator（可选优化）

---

## 测试验证

### 当前测试状态

| 测试项 | 状态 | 说明 |
|-------|------|------|
| 基础 mm/bmm 测试 | ✅ 通过 | 精度误差 < 1e-5 |
| Linear 层测试 | ✅ 通过 | 768→3072 维度 |
| Qwen 0.5B Forward | ✅ 通过 | 完整前向传播 |

### 测试命令

```bash
# 运行 Qwen 模型测试
python test/test_qwen0.5B.py

# 运行自定义算子测试
python test/test_custom_ops.py
```

---

## 相关文件

| 文件 | 说明 |
|-----|------|
| `runtime/cas_npu_runtime.h` | Runtime API 声明 |
| `runtime/cmodel/simulator.cpp` | cmodel CPU 模拟实现 |
| `runtime/fpga/simulator.cpp` | FPGA 实现（待完善） |
| `backend/cas_npu_ops.cpp` | PyTorch 算子注册 |
| `backend/cas_npu_allocator.cpp` | 内存分配器 |
| `test/test_qwen0.5B.py` | Qwen 模型测试 |

---

## 备注

1. **CPU Fallback 是临时方案**：当前大部分算子使用 Device→CPU→计算→Device 模式，这是为了快速验证功能正确性，不是最终方案。

2. **性能优化的关键**：在 NPU Runtime 层原生实现算子，消除 CPU 往返开销。

3. **内存效率**：当 NPU 原生算子完善后，可考虑实现 Caching Allocator 进一步优化内存分配。

4. **多后端支持**：Runtime 层支持 cmodel（调试）和 fpga/asic（生产）两种实现，通过编译选项切换。

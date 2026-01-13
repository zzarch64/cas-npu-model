# Attention Mask 问题分析

## 问题描述

在使用 ECHO-NPU 后端进行 Qwen 模型推理时，虽然 `attention_mask` 被正确传递和扩展，但模型生成的内容与输入 prompt 完全无关。

## 测试结果

### 1. Attention Mask 传递和扩展
- ✅ `attention_mask` 被正确传递到 `model.generate()`
- ✅ 在生成过程中，`attention_mask` 被正确扩展
- ✅ Hook 显示模型 forward 方法接收到了 `attention_mask`

### 2. Attention Mask 对输出的影响
- ✅ 在 forward pass 中，不同的 `attention_mask` 会产生不同的输出 logits（差异 > 3.0）
- ⚠️ 但在生成过程中，有/无 `attention_mask` 的生成结果**完全相同**

### 3. CPU vs ECHO-NPU 对比
- ✅ 在 CPU 上，模型生成与输入相关的内容
- ❌ 在 ECHO-NPU 上，模型生成与输入无关的内容
- ❌ CPU 和 ECHO-NPU 上的 forward pass 输出 logits 差异显著（最大差异 6.19）

## 根本原因分析

问题不在 `attention_mask` 本身，而在 **ECHO-NPU 后端的实现**。

### 关键发现

1. **`masked_fill_` 未被调用**：
   - Hook 显示 `masked_fill_` 在 forward pass 中未被调用
   - 这可能意味着模型使用了其他方式处理 `attention_mask`（如直接乘法）

2. **Forward pass 输出差异**：
   - CPU 和 ECHO-NPU 上的 logits 差异显著
   - 这说明问题在模型计算层面，而不在生成逻辑

3. **生成结果差异**：
   - CPU 上生成的内容与输入相关
   - ECHO-NPU 上生成的内容与输入无关
   - 这进一步确认了问题在 ECHO-NPU 后端实现

## 可能的问题点

### 1. `_copy_from` 实现
- 虽然修复了数据类型不匹配和非 contiguous tensor 的处理
- 但可能在某些边界情况下仍有问题

### 2. `mm` 和 `bmm` 实现
- 这两个算子有自定义实现
- 可能存在问题，导致矩阵乘法结果错误

### 3. `addmm` 使用 fallback
- `addmm` 使用 `cpu_fallback`
- 虽然不应该导致错误，但可能有性能或正确性问题

## 建议的修复方向

1. **验证 `mm` 和 `bmm` 实现**：
   - 创建单元测试，验证 `mm` 和 `bmm` 的输出与 CPU 版本一致
   - 检查数据拷贝和类型处理是否正确

2. **验证 `_copy_from` 实现**：
   - 创建测试，验证各种数据类型和 stride 组合下的数据拷贝正确性
   - 确保没有数据损坏或精度损失

3. **添加调试输出**：
   - 在关键算子中添加调试输出，比较 CPU 和 ECHO-NPU 上的中间结果
   - 定位具体哪个算子导致输出差异

## 当前状态

- ✅ `attention_mask` 被正确传递和扩展
- ✅ `attention_mask` 在 forward pass 中影响输出
- ❌ 模型在 ECHO-NPU 上的输出与 CPU 不一致
- ❌ 生成的内容与输入无关

---

**创建日期**：2026-01-10  
**状态**：调查中

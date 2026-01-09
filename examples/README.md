# CAS-NPU 示例代码

本目录包含 CAS-NPU 扩展的使用示例，展示如何在真实场景中使用自定义设备。

## 📋 示例列表

### 1. `lenet_inference.py` - LeNet 推理示例

**用途**: 演示如何在 CAS-NPU 设备上运行完整的神经网络推理

**特点**:
- 实现完整的 LeNet 网络结构
- 演示 CPU Fallback 机制（未实现的操作自动回退到 CPU）
- 验证 add 操作使用 CAS-NPU 实现

**运行方式**:
```bash
python examples/lenet_inference.py
```

**展示内容**:
- LeNet 前向传播在 CPU 和 CAS-NPU 上的对比
- 输出一致性验证
- add.Tensor 操作验证
- 多操作链式调用

---

### 2. `lenet_training.py` - LeNet 训练示例

**用途**: 演示完整的训练流程，包括反向传播和梯度计算

**特点**:
- 使用 MSELoss 避免 log_softmax 相关问题
- 实现完整的训练循环
- 验证梯度计算和参数更新

**运行方式**:
```bash
python examples/lenet_training.py
```

**展示内容**:
- add 操作的反向传播
- Linear 层的反向传播
- Conv2d 层的反向传播
- 完整的 LeNet 训练流程
- 梯度累积示例

**注意事项**:
- 使用 MSELoss 而不是 CrossEntropyLoss（需要更多 Autograd 注册）

---

### 3. `qwen_model.py` - Qwen 模型示例

**用途**: 验证矩阵乘法（mm）和批量矩阵乘法（bmm）算子实现

**特点**:
- 测试真实的大语言模型（Qwen 0.5B）
- 验证 mm 和 bmm 算子的正确性
- 测试 Transformer 架构的关键操作
- 支持 LoRA 微调训练

**运行方式**:
```bash
# 运行所有测试（包括 LoRA，如果 peft 可用）
python examples/qwen_model.py

# 强制启用 LoRA 测试
python examples/qwen_model.py --lora

# 跳过 LoRA 测试
python examples/qwen_model.py --no-lora
```

**展示内容**:
- 基础 mm 和 bmm 操作测试
- Linear 层测试（使用 mm）
- Qwen 模型加载和前向传播
- LoRA 微调训练示例（可选，需要 peft 库）

**前置条件**:
- 需要安装 `transformers` 库: `pip install transformers`
- LoRA 功能需要 `peft` 库: `pip install peft`
- 需要实现 `mm` 和 `bmm` 算子（在 `cas_npu_ops.cpp` 中）

**注意事项**:
- 首次运行会下载模型权重（约 1GB）
- 模型推理可能需要较长时间
- LoRA 训练会使用较小的 batch size 和序列长度以节省内存

---

## 🚀 快速开始

### 1. 编译扩展

在运行示例之前，需要先编译 C++ 扩展：

```bash
# 在项目根目录
python setup.py build_ext --inplace
```

### 2. 运行示例

```bash
# LeNet 推理示例
python examples/lenet_inference.py

# LeNet 训练示例
python examples/lenet_training.py

# Qwen 模型示例
python examples/qwen_model.py
```

---

## 📝 依赖要求

### 必需依赖
- PyTorch (>= 1.13.0)
- NumPy

### 可选依赖
- `transformers` - 用于 Qwen 模型示例
  ```bash
  pip install transformers
  ```
- `peft` - 用于 LoRA 微调示例
  ```bash
  pip install peft
  ```

---

## 💡 使用建议

1. **学习顺序**: 建议按照 `lenet_inference.py` → `lenet_training.py` → `qwen_model.py` 的顺序学习
2. **理解 CPU Fallback**: 注意观察哪些操作在 CPU 上执行，哪些在 CAS-NPU 上执行
3. **调试技巧**: 如果遇到问题，可以查看 `test/` 目录下的测试文件了解详细实现

---

## 🔗 相关文档

- [主 README](../README.md) - 项目总体介绍
- [测试文档](../test/README.md) - 测试套件说明
- [开发日志](../DEVLOG.md) - 开发过程记录

---

## 📄 许可证

与主项目保持一致。

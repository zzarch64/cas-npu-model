# ECHO-NPU 文档

本目录包含 ECHO-NPU 扩展的完整开发文档。

## 📚 文档结构

### 核心文档

- **[DEVLOG.md](./DEVLOG.md)** - 开发日志
  - 项目概述和开发思路
  - 架构设计
  - 算子实现进度
  - 开发计划和路线图

### 调试文档

- **[debugging.md](./debugging.md)** - 调试指南
  - GDB 调试方法
  - Valgrind 内存调试
  - AddressSanitizer 使用
  - 快速参考

### 问题分析

- **[issues/memory_issues.md](./issues/memory_issues.md)** - 内存问题分析与修复
  - aligned_alloc 问题分析
  - malloc 修复方案
  - 测试验证

- **[issues/attention_mask_issue.md](./issues/attention_mask_issue.md)** - Attention Mask 问题分析
  - 问题描述和测试结果
  - 根本原因分析
  - 修复方向建议

- **[issues/operator_location.md](./issues/operator_location.md)** - 算子问题定位报告
  - 问题定位结果
  - 测试数据分析
  - 修复建议

## 🚀 快速导航

### 新开发者

1. 阅读 [DEVLOG.md](./DEVLOG.md) 了解项目架构和开发思路
2. 查看 [README.md](../README.md) 了解快速开始
3. 遇到问题时参考 [debugging.md](./debugging.md)

### 调试问题

1. 查看 [debugging.md](./debugging.md) 选择合适的调试工具
2. 参考 [issues/](./issues/) 目录下的问题分析文档
3. 使用 GDB 或 AddressSanitizer 进行深入调试

### 开发新功能

1. 参考 [DEVLOG.md](./DEVLOG.md) 中的开发计划和优先级
2. 遵循渐进式开发策略（Fallback → NPU 原生）
3. 添加相应的测试用例

## 📝 文档维护

- 文档按功能分类组织
- 已解决的问题保留在 `issues/` 目录作为参考
- 调试指南统一在 `debugging.md` 中
- 开发日志保持更新，记录重要决策和进展

---

**最后更新**：2026-01-10

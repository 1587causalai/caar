# CAAR 项目文档

欢迎来到 CAAR (Cauchy Abduction Action Regression) 项目的文档站点！

## 项目概述

本项目实现了基于推断/行动框架的新型鲁棒回归模型 CAAR，通过对比不同算法在各种数据条件下的表现，验证了 CAAR 在处理异常值方面的优势。

## 文档导航

### 📋 实验设计
- [实验设计](experiment_design.md) - 详细的实验设计方案
- [实验方法](experiment_method.md) - 实验方法论和技术细节
- [网络设计](network_design.md) - 神经网络架构设计

### 📊 实验报告
- [合成数据实验报告](synthetic_exp_report.md) - 合成数据集上的实验结果
- [真实数据实验报告](real_exp_report.md) - 真实数据集上的实验结果
- [实验总结](experiment_summary.md) - 实验结果汇总分析

### 📈 结论与分析
- [实验总结](experiment_summary.md) - 实验结果汇总分析

## 快速开始

1. 从左侧导航栏选择感兴趣的章节
2. 每个章节都包含详细的分析和可视化结果
3. 图表和数据可在 `images/` 目录中找到

## 技术栈

- **基线算法**: OLS, Random Forest, XGBoost
- **神经网络方法**: MLP (MSE), MLP_Huber, MLP_Pinball_Median, MLP_Cauchy
- **创新方法**: CAAR (柯西推断行动回归), GAAR (高斯推断行动回归)
- **评估指标**: MSE, MAE, RMSE, MdAE, R²
- **可视化**: Python matplotlib, seaborn
- **文档**: Docsify

---

*本文档使用 [Docsify](https://docsify.js.org/) 构建* 
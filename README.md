# 项目说明

## 项目概述

本项目实现了基于推断/行动(Abduction/Action)的新型回归模型（CAAR: Cauchy Abduction Action Regression），并通过一系列实验验证了该模型在处理含异常点数据时的鲁棒性优势。项目包含完整的模型实现、实验代码、数据处理模块、评估工具和详细的实验报告。

## 项目结构

```
robust-regression-experiment/
├── src/                      # 源代码目录
│   ├── models/               # 模型实现
│   │   ├── caar.py           # CAAR模型实现
│   │   ├── baseline.py       # 基线模型实现
│   │   └── robust.py         # 现有鲁棒回归方法实现
│   ├── data/                 # 数据处理模块
│   │   ├── synthetic.py      # 合成数据生成
│   │   └── real.py           # 真实数据加载
│   ├── utils/                # 工具模块
│   │   ├── metrics.py        # 评估指标
│   │   ├── visualization.py  # 可视化工具
│   │   └── document.py       # 文档生成
│   ├── experiments/          # 实验执行模块
│   │   ├── synthetic_exp.py  # 合成数据实验
│   │   └── real_exp.py       # 真实数据实验
│   └── run_experiments.py    # 主实验执行脚本
├── data/                     # 数据目录
├── results/                  # 实验结果目录
│   ├── tables/               # 性能表格
│   └── figures/              # 可视化图表
├── docs/                     # 文档目录
│   ├── experiment_summary.md # 实验结果总结
│   ├── experiment_method.md  # 实验方法详解
│   ├── experiment_conclusion.md # 实验结论分析
│   ├── full_report.md        # 完整实验报告
│   └── images/               # 文档引用图片
├── requirements.txt          # 项目依赖
└── README.md                 # 项目说明
```

## 安装与运行

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- scikit-learn 1.3.0+
- pandas 2.0+
- numpy 1.24+
- matplotlib 3.7+
- seaborn 0.12+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行实验

```bash
# 运行所有实验
python src/run_experiments.py

# 仅运行合成数据实验
python src/experiments/synthetic_exp.py

# 仅运行真实数据实验
python src/experiments/real_exp.py
```

## 实验报告

详细的实验报告位于`docs/`目录下：

- `experiment_summary.md`: 实验结果总结
- `experiment_method.md`: 实验方法详解
- `experiment_conclusion.md`: 实验结论分析
- `full_report.md`: 完整实验报告

## 主要发现

通过对合成数据和真实数据的全面实验，我们验证了CAAR模型在处理含异常点数据时的卓越性能：

1. **优异的鲁棒性**：CAAR模型在各种异常值场景下都表现出色，随着异常值比例的增加，其性能下降幅度明显小于传统方法。

2. **广泛的适用性**：CAAR模型不仅适用于线性关系，在非线性关系和复杂的真实世界数据中同样能够有效工作。

3. **稳定的预测**：CAAR模型在中位数绝对误差（MdAE）指标上表现尤为突出，这表明其预测结果更加稳定可靠。

4. **计算效率**：CAAR模型在保持高鲁棒性的同时，计算效率也较为理想，特别是与其他复杂的鲁棒方法相比。

## 联系方式

如有任何问题或建议，请联系项目作者。

# 项目说明(废弃)


> 当前项目内容已经演化成为 https://github.com/1587causalai/causal-sklearn


## 项目概述

本项目实现了基于推断/行动(Abduction/Action)的新型回归模型（CAAR: Cauchy Abduction Action Regression），并通过一系列实验验证了该模型在处理含异常点数据时的鲁棒性优势。项目包含完整的模型实现、实验代码、数据处理模块、评估工具和详细的实验报告。

## 项目结构

```
robust-regression-experiment/
├── src/                      # 源代码目录
│   ├── models/               # 模型实现
│   ├── data/                 # 数据处理模块
│   ├── utils/                # 工具模块
│   ├── experiments/          # 实验执行模块
│   └── run_experiments.py    # 主实验执行脚本
├── data/                     # 数据目录
│   ├── Bike-Sharing-Dataset.zip
│   ├── communities.data
│   ├── communities.names
│   ├── Concrete_Data.xls
│   ├── hour.csv
│   ├── parkinsons_updrs.data
│   └── winequality-red.csv
├── results/                  # 实验结果目录
│   ├── all_real_results.pkl
│   ├── all_synthetic_results.pkl
│   ├── experiment_log.txt
│   ├── tables/               # 性能表格
│   ├── real_*_outliers/      # 真实数据实验结果
│   └── synthetic_*_outliers/ # 合成数据实验结果
├── docs/                     # 文档目录
│   ├── _navbar.md
│   ├── _sidebar.md
│   ├── index.html
│   ├── experiment_design.md  # 实验设计方案
│   ├── experiment_method.md  # 实验方法详解
│   ├── experiment_summary.md # 实验结果总结
│   ├── network_design.md     # 神经网络架构设计
│   ├── real_exp_report.md    # 真实数据实验报告
│   ├── synthetic_exp_report.md # 合成数据实验报告
│   ├── images/               # 文档引用图片
│   └── README.md
├── convert_to_html_img.py    # 图片路径转换工具
├── download_data.py          # 数据下载脚本
├── serve_docs.py             # 文档服务器
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

## 📚 文档与报告

### 在线文档站点

我们提供了完整的在线文档站点，包含详细的实验报告和可视化结果：

- **在线访问**: [项目文档站点](https://1587causalai.github.io/caar)
- **本地预览**: `python serve_docs.py`

### 文档内容

详细的实验报告位于`docs/`目录下：

- `experiment_design.md`: 实验设计方案
- `experiment_method.md`: 实验方法详解
- `network_design.md`: 神经网络架构设计
- `synthetic_exp_report.md`: 合成数据实验报告
- `real_exp_report.md`: 真实数据实验报告
- `experiment_summary.md`: 实验结果总结
- `experiment_conclusion.md`: 实验结论分析

### 本地文档服务

```bash
# 启动文档服务器（推荐）
python serve_docs.py

# 或使用 docsify-cli
npm install -g docsify-cli
docsify serve docs

# 或使用 Python 内置服务器
cd docs && python -m http.server 3000
```

### 部署到GitHub Pages

由于文档中引用了大量实验结果图片，部署前需要运行部署脚本：

```bash
# 运行部署脚本，将图片复制到docs目录
./deploy_docs.sh

# 然后提交并推送
git add docs/results
git commit -m "Add results images for GitHub Pages"
git push
```

## 主要发现

通过对合成数据和真实数据的全面实验，我们验证了CAAR模型在处理含异常点数据时的卓越性能：

1. **优异的鲁棒性**：CAAR模型在各种异常值场景下都表现出色，随着异常值比例的增加，其性能下降幅度明显小于传统方法。

2. **广泛的适用性**：CAAR模型不仅适用于线性关系，在非线性关系和复杂的真实世界数据中同样能够有效工作。

3. **稳定的预测**：CAAR模型在中位数绝对误差（MdAE）指标上表现尤为突出，这表明其预测结果更加稳定可靠。

4. **计算效率**：CAAR模型在保持高鲁棒性的同时，计算效率也较为理想，特别是与其他复杂的鲁棒方法相比。

## 引用

如果您在研究中使用了本项目的CAAR模型或相关代码，请引用我们的工作：

```bibtex
@misc{caar2025,
  title={CAAR: Cauchy Abduction Action Regression for Robust Regression with Outliers},
  author={Heyang Gong},
  year={2025},
  howpublished={\url{https://github.com/1587causalai/robust-regression-experiment}},
  note={GitHub repository}
}
```

## 联系方式

如有任何问题或建议，请联系项目作者。

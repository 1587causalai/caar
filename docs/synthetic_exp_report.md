# CAAR 模型鲁棒性：合成数据实验报告

## 1. 引言与背景

本文档详细介绍了在合成数据集上评估柯西推断行动回归（CAAR）模型性能和鲁棒性的实验设置与结果。与真实数据实验相比，合成数据实验允许我们更精确地控制数据生成过程、异常值的类型和强度，从而更细致地分析模型在特定条件下的行为。

这些实验的核心目标是：
1.  验证 CAAR 模型在不同类型数据关系（线性和非线性）下对Y轴异常值的鲁棒性。
2.  与一系列基线模型和鲁棒回归技术进行性能比较。
3.  理解数据特性（如关系复杂性、异常值强度）如何影响各模型的表现。

实验遵循"训练集包含异常点，测试集保持纯净"的原则，以公正评估模型从受污染数据到干净数据的泛化能力。

## 2. 对比方法概述

实验中，CAAR 及 GAAR 方法与多种回归模型进行了对比，这些模型覆盖了常见基线、现有鲁棒方法以及与我们提出的新方法结构相似的神经网络模型。

*   **常见基线模型与神经网络对比模型：**
    *   **OLS (普通最小二乘法):** 作为基础线性模型基准（主要用于线性场景）。
    *   **RandomForest (随机森林回归):** 代表集成学习方法。
    *   **XGBoost (XGBoost 回归):** 高性能梯度提升集成方法。
    *   **MLP (多层感知机回归 - MSE损失):** 标准神经网络模型，使用均方误差(MSE)损失，作为其他基于MLP的鲁棒方法的基础对比。
*   **基于神经网络的鲁棒回归模型：**
    *   **MLP_Huber (多层感知机回归 - Huber损失):** 与MLP结构相似，但使用Huber损失函数（实验中设置delta=1.35），旨在平衡MSE和MAE的特性以抵抗异常值。
    *   **MLP_Pinball_Median (多层感知机回归 - Pinball损失实现中位数回归):** 与MLP结构相似，但使用Pinball损失函数（quantile=0.5）来拟合条件中位数，旨在通过优化中位数目标提升鲁棒性。
*   **我们的新方法（基于推断/行动框架）：**
    *   **CAAR (柯西推断行动回归):** 通过推断网络为每个样本推断潜在子群体的柯西分布，并利用行动网络进行回归。柯西分布的重尾特性使其对异常值具有天然的鲁棒性。
    *   **GAAR (高斯推断行动回归):** 与CAAR结构相似，但推断网络基于高斯分布假设，用作对比分析分布假设对鲁棒性的影响。

这个模型组合有助于全面评估 CAAR 在不同合成数据场景下的相对优势和劣势，特别是与其他基于相似神经网络架构但采用不同鲁棒策略的模型进行比较。

## 3. 实验设置详解

实验通过 `src/experiments/synthetic_exp.py` 脚本执行。

### 3.1 数据生成

合成数据由 `src/data/synthetic.py` 中的 `prepare_synthetic_experiment` 函数生成。实验主要涵盖以下两种数据生成场景：

1.  **线性关系 (`relation_type='linear'`):**
    *   数据通过 `make_regression` 生成，具有固定的特征数量。
    *   目标 `y` 与特征 `X` 之间存在线性关系。
2.  **非线性关系 (`relation_type='interactive_heteroscedastic'`):**
    *   数据通过自定义函数 `make_interactive_heteroscedastic_regression` 生成。
    *   此场景引入了特征间的交互项以及异方差性（误差方差随X变化），更接近真实世界数据的复杂性。
    *   具体函数形式为：`y = 10 * sin(X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5)**2 + 10 * X[:, 3] + 5 * X[:, 4] + error`，其中 error 项具有异方差性。

对于每个场景：
*   生成包含 `n_samples_train` (例如 2000), `n_samples_val` (例如 500), 和 `n_samples_test` (例如 500) 的样本。
*   特征数量 `n_features` (例如 10) 和信息特征数量 `n_informative` (例如 5，仅用于线性场景) 被设定。
*   使用固定的 `random_state` 以确保数据生成和划分的可复现性。
*   **关键点：测试集 (`X_test`, `y_test`) 在任何异常值注入之前被分离出来，并保持纯净。**

### 3.2 数据预处理与划分

1.  **标准化:** 特征数据 (`X`) 使用 `sklearn.preprocessing.StandardScaler` 进行标准化。标准化器在完整特征集上拟合，然后在划分前应用于所有数据。
2.  **划分:** 数据按指定比例划分为训练集、验证集和测试集。

### 3.3 异常值注入

异常值**仅注入训练集和验证集**，通过 `src/data/synthetic.py` 中的 `inject_y_outliers` 函数完成。

*   **控制参数:**
    *   `outlier_ratio`: 转化为异常样本的比例 (例如, 0.1, 0.2)。
    *   `outlier_strength`: 一个乘数（实验中通常为 5.0），决定异常值偏离正常值的程度（相对于原始 `y` 值的标准差）。
    *   `y_outlier_method`: 实验中采用 `'sequential_multiplicative_additive'` 方法。该方法结合了乘性和加性噪声来生成异常值，使其更难被简单阈值检测。
        *   一半异常点：`y_outlier = y_normal * (1 + strength) + N(0, global_std_y * 0.1 * strength)`
        *   另一半异常点：`y_outlier = y_normal * (1 - strength) - N(0, global_std_y * 0.1 * strength)`

### 3.4 模型参数与训练

*   **OLS:** 使用 scikit-learn 的默认参数。
*   **RandomForest, XGBoost:** 使用实验脚本中指定的参数 (例如, `n_estimators=100` for RandomForest)。
*   **神经网络模型 (CAAR, GAAR, MLP, MLP_Pinball_Median, MLP_Huber):**
    *   共享参数 (`nn_model_params`):
        *   `input_dim`: 根据数据集动态设置。
        *   `hidden_dims`: `[64, 32]`。
        *   `epochs`: `100`。
        *   `lr`: `0.001`。
        *   `batch_size`: `32`。
        *   `early_stopping_patience`: `10`。
        *   `early_stopping_min_delta`: `0.0001`。
    *   特定参数:
        *   `CAAR` 和 `GAAR`: `latent_dim: 16`。
        *   `MLP_Pinball_Median`: `quantile: 0.5`。
        *   `MLP_Huber`: `delta: 1.35`。
*   模型在（可能受异常值污染的）训练集上训练，验证集用于早停。

### 3.5 重复与评估

*   **重复次数:** 每个实验条件重复 `n_repeats` 次（例如3次），使用不同的随机种子。
*   **评估指标:** 模型在**纯净的测试集**上使用以下指标进行评估：MSE, RMSE, MAE, MdAE, R² 和训练时间。
*   报告结果是这些指标在多次重复中的平均值。

## 4. 实验结果与讨论

本节展示并分析合成数据实验生成的性能表格。

### 4.1 线性关系 - Y轴异常 (Linear Relation - Y-axis Outliers)

下表展示了在线性关系合成数据中注入Y轴异常值（强度5.0，`sequential_multiplicative_additive`方法）后的平均性能：

| Model              |        MSE |      RMSE |      MAE |     MdAE |        R² |   Training Time (s) |
|:-------------------|-----------:|----------:|---------:|---------:|----------:|--------------------:|
| OLS                |  17.3983   |  3.48861  | 2.76629  | 2.35207  |  0.832716 |          0.00815158 |
| MLP_Huber          |   0.260948 |  0.5108   | 0.405114 | 0.335928 |  0.997464 |          0.0634839  |
| RandomForest       |  65.5703   |  8.08342  | 6.32403  | 5.27264  |  0.36723  |         20.9788     |
| XGBoost            |  49.316    |  7.01617  | 5.51229  | 4.51361  |  0.521519 |          0.135208   |
| CAAR               |   0.652868 |  0.80676  | 0.643363 | 0.542435 |  0.993655 |          1.92835    |
| MLP                | 239.236    | 12.4544   | 9.72582  | 8.02128  | -1.28284  |          0.597448   |
| GAAR               |  42.632    |  5.20301  | 4.10131  | 3.37564  |  0.595151 |          1.56069    |
| MLP_Pinball_Median |   0.918042 |  0.940916 | 0.74535  | 0.622694 |  0.99109  |          0.953271   |

*数据来源: `results/synthetic_linear_y_outliers/performance_table.md`*

**结果分析 (线性关系 - Y轴异常):**
*   **MLP_Huber表现最佳：** 在线性关系且存在Y轴异常的场景下，**MLP_Huber (MSE 0.26, R² 0.9975)** 表现最为出色，其MSE显著低于其他模型，R²非常接近1。
*   **CAAR和MLP_Pinball_Median表现优异：** **CAAR (MSE 0.65, R² 0.9937)** 和 **MLP_Pinball_Median (MSE 0.92, R² 0.9911)** 的性能也非常高，R²值均超过0.99，显示了它们强大的鲁棒性。MdAE指标也较低，分别为0.54和0.62。
*   **OLS受影响但仍具一定预测力：** OLS (MSE 17.40, R² 0.8327) 受到异常值影响，MSE较高，但R²仍为正，表明在线性数据和当前异常强度下，它仍保留了一定的预测能力。
*   **其他模型表现不佳：** RandomForest (R² 0.3672), XGBoost (R² 0.5215), 和 GAAR (R² 0.5952) 的R²值相对较低。标准MLP (R² -1.2828) 在此场景下完全失效，R²为负。
*   **训练时间：** OLS训练最快。MLP_Huber, XGBoost, MLP, GAAR, MLP_Pinball_Median, CAAR的训练时间均在2秒以内。RandomForest训练时间最长（约21秒）。

**相关图表 (线性关系 - Y轴异常):**
*   MSE 性能对比: ![MSE Performance Comparison](../results/synthetic_linear_y_outliers/performance_comparison_MSE.png)
*   MdAE 性能对比: ![MdAE Performance Comparison](../results/synthetic_linear_y_outliers/performance_comparison_MdAE.png)
*   R² 性能对比: ![R2 Performance Comparison](../results/synthetic_linear_y_outliers/performance_comparison_R².png)
*   MSE 趋势图: ![MSE Trend vs Outlier Ratio](../results/synthetic_linear_y_outliers/trend_MSE.png)
*   MdAE 趋势图: ![MdAE Trend vs Outlier Ratio](../results/synthetic_linear_y_outliers/trend_MdAE.png)
*   R² 趋势图: ![R2 Trend vs Outlier Ratio](../results/synthetic_linear_y_outliers/trend_R².png)

### 4.2 非线性关系 ('interactive_heteroscedastic') - Y轴异常

下表展示了在非线性（含交互项和异方差性）合成数据中注入Y轴异常值（强度5.0，`sequential_multiplicative_additive`方法）后的平均性能：

| Model              |     MSE |    RMSE |     MAE |    MdAE |        R² |   Training Time (s) |
|:-------------------|--------:|--------:|--------:|--------:|----------:|--------------------:|
| RandomForest       | 26.2723 | 5.11894 | 3.998   | 3.27611 |  0.251473 |           24.487    |
| MLP_Huber          | 17.8906 | 4.21121 | 3.28505 | 2.68828 |  0.495017 |            0.065609 |
| XGBoost            | 20.7258 | 4.5442  | 3.56824 | 2.97075 |  0.409529 |            0.395092 |
| CAAR               | 11.7229 | 3.41687 | 2.71219 | 2.25264 |  0.665724 |            1.44184  |
| MLP                | 72.4173 | 7.77325 | 6.07633 | 4.98829 | -1.08573  |            1.29547  |
| GAAR               | 53.1069 | 6.71068 | 5.19822 | 4.19647 | -0.527773 |            1.41482  |
| MLP_Pinball_Median | 20.2187 | 4.3329  | 3.40841 | 2.77901 |  0.423891 |            2.04047  |

*数据来源: `results/synthetic_nonlinear_y_outliers/performance_table.md`*

**结果分析 (非线性关系 - Y轴异常):**
*   **CAAR表现最佳：** 在更复杂的非线性（含交互项和异方差性）数据且存在Y轴异常的场景下，**CAAR (MSE 11.72, R² 0.6657)** 表现最佳，其MSE最低，R²最高。MdAE (2.25) 也非常低。
*   **MLP_Huber次之：** **MLP_Huber (MSE 17.89, R² 0.4950)** 的表现次于CAAR，但仍然展示了较好的鲁棒性。
*   **MLP_Pinball_Median和XGBoost表现尚可：** **MLP_Pinball_Median (MSE 20.22, R² 0.4239)** 和 **XGBoost (MSE 20.73, R² 0.4095)** 的R²为正，但性能不如CAAR和MLP_Huber。
*   **RandomForest表现一般：** RandomForest (MSE 26.27, R² 0.2515) 在此场景下表现一般。
*   **MLP和GAAR失效：** 标准MLP (R² -1.0857) 和 GAAR (R² -0.5278) 的R²为负，表明它们无法处理这种复杂的非线性数据和Y轴异常的组合。OLS在此非线性场景中未作为主要对比（通常预期其表现会很差）。
*   **训练时间：** MLP_Huber训练最快。CAAR, MLP, GAAR, MLP_Pinball_Median, XGBoost的训练时间均在2秒左右或以内。RandomForest训练时间最长（约24秒）。


**相关图表 (非线性关系 - Y轴异常):**
*   MSE 性能对比: ![MSE Performance Comparison](../results/synthetic_nonlinear_y_outliers/performance_comparison_MSE.png)
*   MdAE 性能对比: ![MdAE Performance Comparison](../results/synthetic_nonlinear_y_outliers/performance_comparison_MdAE.png)
*   R² 性能对比: ![R2 Performance Comparison](../results/synthetic_nonlinear_y_outliers/performance_comparison_R².png)
*   MSE 趋势图: ![MSE Trend vs Outlier Ratio](../results/synthetic_nonlinear_y_outliers/trend_MSE.png)
*   MdAE 趋势图: ![MdAE Trend vs Outlier Ratio](../results/synthetic_nonlinear_y_outliers/trend_MdAE.png)
*   R² 趋势图: ![R2 Trend vs Outlier Ratio](../results/synthetic_nonlinear_y_outliers/trend_R².png)

## 5. 初步结论与展望

基于对两种合成数据集（线性关系、非线性交互异方差关系）在Y轴异常（强度5.0，`sequential_multiplicative_additive`方法）场景下的实验结果，我们可以得出以下初步结论：

*   **CAAR在非线性数据上展现优势：** 对于具有复杂非线性关系和异方差性的数据，CAAR模型表现出最佳的鲁棒性和预测精度，在MSE、R²和MdAE等关键指标上均领先。
*   **MLP_Huber在线性数据上表现出色：** 对于线性关系数据，基于神经网络的MLP_Huber回归展现了极佳的鲁棒性，性能甚至略优于CAAR和MLP_Pinball_Median。这表明对于相对简单的数据结构，使用Huber损失的MLP模型非常有效。
*   **MLP_Pinball_Median作为通用鲁棒方法：** MLP_Pinball_Median在两种场景下均表现出良好的鲁棒性（线性R² > 0.99，非线性R² > 0.42），尽管在非线性场景不如CAAR，在线性场景不如MLP_Huber，但其通用性值得肯定。
*   **模型对数据复杂性的敏感度：**
    *   OLS仅在线性数据上（无异常或弱异常时）有效。
    *   标准MLP和GAAR在两种合成数据场景下，当Y轴异常存在时，表现均不佳，尤其是在非线性场景下R²为负。这可能表明它们的标准损失函数或结构不足以应对具有挑战性的异常值和数据复杂性。
    *   RandomForest和XGBoost等集成方法在非线性数据上有一定表现，但鲁棒性不如专门设计的鲁棒模型（如CAAR, MLP_Huber）。

*   **未来工作展望：**
    *   **不同强度和比例的异常值：** 系统研究不同`outlier_strength`和`outlier_ratio`对各模型性能的影响。
    *   **X轴异常值实验：** 如`synthetic_exp.py`中预留，执行并分析X轴异常（杠杆点）对模型性能的影响。
    *   **更多非线性场景：** 探索更多不同类型的非线性关系和噪声结构。
    *   **超参数敏感性分析：** 针对表现较好的模型（如CAAR, MLP_Huber, MLP_Pinball_Median），研究其关键超参数对鲁棒性的影响。

总而言之，合成数据实验为我们提供了受控环境下评估模型鲁棒性的宝贵机会。结果表明，CAAR在处理复杂非线性数据中的Y轴异常方面具有潜力，而MLP_Huber和MLP_Pinball_Median也是非常值得关注的鲁棒方法。

---
*本报告基于对提供的Python脚本和性能表格数据的分析。通过可视化生成的图表和检查 `full_results.pkl` 中的原始结果，可能会获得更深入的见解。* 
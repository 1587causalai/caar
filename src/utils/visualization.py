"""
可视化工具模块

提供以下功能：
1. 绘制性能对比表格
2. 绘制不同异常值比例下的性能趋势图
3. 绘制残差分析图
4. 绘制模型预测与真实值对比图
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

def plot_performance_comparison(results_df, metric, title_prefix="", save_path=None, show_plot=True):
    """
    绘制不同模型在指定指标上的性能对比图（条形图）。

    参数:
        results_df: 包含模型性能数据的DataFrame
        metric: 要绘制的性能指标名称 (例如 'MSE_mean', 'R²_mean')
        title_prefix: 图像标题前缀
        save_path: 图像保存路径 (如果为None则不保存)
        show_plot: 是否显示图像 (默认为True)
    """
    try:
        plt.figure(figsize=(12, 7))
        
        # 确保metric列存在
        if metric not in results_df.columns:
            print(f"错误：指标 '{metric}' 不在DataFrame中。可用指标: {results_df.columns.tolist()}")
            return

        # 排序以便更好地可视化
        sorted_df = results_df.sort_values(by=metric, ascending=True)
        
        bars = plt.barh(sorted_df.index, sorted_df[metric], color=plt.cm.viridis(np.linspace(0, 1, len(sorted_df.index))))
        
        plt.xlabel(f'{metric.replace("_mean", "")} Value')
        plt.ylabel('Model')
        plt.title(f'{title_prefix}Model Performance Comparison: {metric.replace("_mean", "")}')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 在条形图上添加数值标签
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                     f'{bar.get_width():.4f}',
                     va='center', ha='left', fontsize=9)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close() # 如果不显示，则关闭图像以释放内存

    except Exception as e:
        print(f"Error during plotting performance comparison: {e}")
        if save_path: # 尝试关闭未完成的图
            try:
                plt.close()
            except:
                pass

def plot_trend_with_outlier_ratio(trend_df, metric, title_prefix="", save_path=None, show_plot=True):
    """
    绘制不同模型在指定指标上随异常值比例变化的趋势图。

    参数:
        trend_df: 包含趋势数据的DataFrame (列: '异常值比例', Model1, Model2, ...)
        metric: 要绘制的性能指标名称 (例如 'RMSE', 'MAE')
        title_prefix: 图像标题前缀
        save_path: 图像保存路径 (如果为None则不保存)
        show_plot: 是否显示图像 (默认为True)
    """
    try:
        plt.figure(figsize=(12, 7))
        
        models = [col for col in trend_df.columns if col != 'Outlier Ratio'] # 确保使用英文列名
        x_axis_label = 'Outlier Ratio'
        
        if x_axis_label not in trend_df.columns:
            print(f"Error: Column '{x_axis_label}' not found in trend_df. Available columns: {trend_df.columns.tolist()}")
            # 尝试兼容旧的中文列名，但发出警告
            if '异常值比例' in trend_df.columns:
                print("Warning: Using legacy column name '异常值比例'. Please update to 'Outlier Ratio'.")
                x_values = trend_df['异常值比例']
            else:
                plt.close()
                return
        else:
            x_values = trend_df[x_axis_label]

        for model_name in models:
            if model_name in trend_df:
                 plt.plot(x_values, trend_df[model_name], marker='o', linestyle='-', label=model_name)
            else:
                print(f"Warning: Model {model_name} not found in trend_df columns for metric {metric}.")
        
        plt.xlabel(x_axis_label)
        plt.ylabel(metric)
        plt.title(f'{title_prefix}Performance Trend vs. Outlier Ratio: {metric}')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        print(f"Error during plotting trend with outlier ratio: {e}")
        if save_path:
            try:
                plt.close()
            except:
                pass

def plot_residuals(y_true, y_pred, model_name_info, save_path=None, show_plot=True):
    """
    绘制残差分析图，包括残差散点图和残差直方图。

    参数:
        y_true: 真实值
        y_pred: 预测值
        model_name_info: 模型名称和附加信息 (例如 'OLS (Outlier Ratio=0.1)')
        save_path: 图像保存路径 (如果为None则不保存)
        show_plot: 是否显示图像 (默认为True)
    """
    try:
        residuals = np.array(y_true).flatten() - np.array(y_pred).flatten() #确保y_true和y_pred是一维的
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 残差散点图
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='w', linewidth=0.5)
        axes[0].axhline(0, color='red', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residual Plot for {model_name_info}')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # 残差直方图
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residual Distribution for {model_name_info}')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        fig.suptitle(f'Residual Analysis for {model_name_info}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应总标题
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"Error during plotting residuals: {e}")
        if save_path:
            try:
                plt.close()
            except:
                pass

def plot_prediction_vs_true(y_true, y_pred, model_name_info, save_path=None, show_plot=True):
    """
    绘制预测值与真实值的对比图。

    参数:
        y_true: 真实值
        y_pred: 预测值
        model_name_info: 模型名称和附加信息 (例如 'OLS (Outlier Ratio=0.1)')
        save_path: 图像保存路径 (如果为None则不保存)
        show_plot: 是否显示图像 (默认为True)
    """
    try:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
        
        # 绘制理想的 y=x 线
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit (y=x)')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Prediction vs. True Values for {model_name_info}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal') # 确保x和y轴的比例相同
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    except Exception as e:
        print(f"Error during plotting prediction vs true: {e}")
        if save_path:
            try:
                plt.close()
            except:
                pass
                
# --- 表格创建和格式化 ---
def create_performance_table(model_results):
    """
    根据模型结果创建性能对比表格。

    参数:
        model_results: 字典，键为模型名称，值为包含指标均值的字典。
                       例如: {'OLS': {'MSE_mean': 0.5, 'R²_mean': 0.9, '训练时间(秒)': 0.1}, ...}
                       或者 {'OLS': {'MSE_mean': 0.5, 'R²_mean': 0.9, 'Training Time (s)': 0.1}, ...}


    返回:
        results_df: Pandas DataFrame，包含格式化后的性能数据。
    """
    # 重命名中文列名为英文
    processed_model_results = {}
    for model_name, metrics in model_results.items():
        processed_metrics = {}
        for metric_name, value in metrics.items():
            if metric_name == '训练时间(秒)' or metric_name == 'Training Time (s)': # 兼容旧key
                processed_metrics['Training Time (s)'] = value
            elif metric_name.endswith('_mean'):
                processed_metrics[metric_name.replace('_mean','')] = value # 移除 _mean 后缀
            else:
                processed_metrics[metric_name] = value
        processed_model_results[model_name] = processed_metrics
        
    results_df = pd.DataFrame.from_dict(processed_model_results, orient='index')
    
    # 定义期望的列顺序 (英文)
    expected_columns_ordered = [
        'MSE', 'RMSE', 'MAE', 'MdAE', 'R²', 'Training Time (s)'
    ]
    
    # 获取实际存在的列，并按照期望顺序排序
    present_columns = [col for col in expected_columns_ordered if col in results_df.columns]
    # 添加任何不在期望列表中的其他列（以防万一）
    other_columns = [col for col in results_df.columns if col not in present_columns]
    final_columns = present_columns + other_columns
    
    results_df = results_df[final_columns]
    
    return results_df

def format_performance_table_markdown(results_df):
    """
    将性能DataFrame格式化为Markdown表格。

    参数:
        results_df: Pandas DataFrame，包含性能数据。

    返回:
        markdown_table: 字符串形式的Markdown表格。
    """
    # 确保索引有名称，以便在Markdown表格中作为第一列的标题
    if results_df.index.name is None:
        results_df.index.name = "Model"
        
    markdown_table = results_df.to_markdown()
    return markdown_table

# 设置全局字体以支持中文（如果需要，但目标是全英文图表）
# try:
#     # 例如: 使用 'SimHei' 字体，如果系统中有的话
#     # font_path = '/System/Library/Fonts/Supplemental/SimHei.ttf' # macOS 示例
#     # my_font = FontProperties(fname=font_path)
#     # mpl.rcParams['font.family'] = my_font.get_name()
#     mpl.rcParams['axes.unicode_minus'] = False # 正确显示负号
# except:
#     print("自定义中文字体未设置，部分中文可能无法正常显示。")
#     mpl.rcParams['axes.unicode_minus'] = False # 正确显示负号

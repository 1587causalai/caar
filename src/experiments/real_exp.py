"""
真实数据实验执行模块

执行以下实验：
1. 原始真实数据实验
2. 注入异常值的真实数据实验
"""

import numpy as np
import pandas as pd
import time
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入自定义模块
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

from models.caar import CAARModel, MLPModel, GAARModel, MLPPinballModel, MLPHuberModel
from models.baseline import (
    OLSRegressor, 
    RidgeRegressor, 
    RandomForestRegressorWrapper,
    XGBoostRegressorWrapper
)
from data.real import prepare_real_data_experiment
from utils.metrics import evaluate_model
from utils.visualization import (
    plot_performance_comparison, 
    plot_trend_with_outlier_ratio, 
    plot_residuals, 
    plot_prediction_vs_true,
    create_performance_table,
    format_performance_table_markdown
)

def run_real_data_experiment(
    dataset_name='california',
    outlier_ratios=  [0.0], # [0.0, 0.05, 0.1, 0.2], 
    outlier_strength=5.0, # 异常值强度
    outlier_type='y',
    n_repeats=1,
    random_state=42,
    results_dir=DEFAULT_RESULTS_DIR,
    nn_early_stopping_patience=10,
    y_outlier_method='additive_std'
):
    """
    执行真实数据实验
    
    参数:
        dataset_name: 数据集名称，可选值为 'california', 'diabetes'
        outlier_ratios: 异常值比例列表
        outlier_strength: 异常值强度
        outlier_type: 异常值类型，可选值为 'y', 'x', 'none'
        n_repeats: 重复次数
        random_state: 随机种子
        results_dir: 结果保存目录
        nn_early_stopping_patience: 神经网络模型早停的耐心轮数
        y_outlier_method: 当outlier_type='y'时，注入y异常值的方法。
        
    返回:
        results: 实验结果字典
    """
    print(f"开始执行{dataset_name}数据集 + {outlier_type}异常值实验...")
    
    # 创建结果保存目录
    experiment_name_suffix = f'{dataset_name}_{outlier_type}_outliers'
    experiment_dir = os.path.join(results_dir, f'real_{experiment_name_suffix}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 初始化结果字典
    results = {
        'outlier_ratios': outlier_ratios,
        'models': {},
        'metrics': ['MSE', 'RMSE', 'MAE', 'MdAE', 'R²'],
        'train_times': {},
        'trend_data': {}
    }
    
    # 初始化模型字典
    models = {
        'OLS': OLSRegressor(),
        # 'Ridge': RidgeRegressor(alpha=1.0), # 和 OLS 效果差不多，没必要用
        'RandomForest': RandomForestRegressorWrapper(n_estimators=100, random_state=random_state),
        'XGBoost': XGBoostRegressorWrapper(random_state=random_state),
        # 'LightGBM': LightGBMRegressorWrapper(random_state=random_state), # 有了 XGBoost 了，没必要用 LightGBM
        # 'QuantileRegressor': QuantileRegressorWrapper(quantile=0.5), # Replaced by MLPPinball_Median
        # 'RANSAC': RANSACRegressorWrapper(), # 效果太差，不配其他算法比
        'MLP': None,
        'MLP_Huber': None, # Placeholder for MLPHuberModel
        'MLP_Pinball_Median': None, # Added MLP Pinball Median placeholder
        'GAAR': None,
        'CAAR': None
    }
    
    # 为每个异常值比例初始化结果
    for ratio in outlier_ratios:
        results['trend_data'][ratio] = {}
        for model_name in models.keys():
            results['trend_data'][ratio][model_name] = {}
    
    # 为每个模型初始化结果
    for model_name in models.keys():
        results['models'][model_name] = {metric: [] for metric in results['metrics']}
        results['train_times'][model_name] = []
    
    # 对每个异常值比例进行实验
    for ratio in outlier_ratios:
        print(f"\n异常值比例: {ratio}")
        
        # 重复实验多次以获得统计可靠性
        for repeat in range(n_repeats):
            print(f"  重复 {repeat+1}/{n_repeats}")
            
            current_seed = random_state + repeat
            X_train, X_val, X_test, y_train, y_val, y_test, feature_names, outlier_mask_train, outlier_mask_val, scaler = prepare_real_data_experiment(
                dataset_name=dataset_name,
                outlier_ratio=ratio,
                outlier_strength=outlier_strength,
                outlier_type=outlier_type,
                y_outlier_method=y_outlier_method,
                random_state=current_seed
            )
            
            input_dim = X_train.shape[1]
            nn_model_params = {
                'input_dim': input_dim, 
                'hidden_dims': [128, 64], 
                'epochs': 100, 
                'lr': 0.001, 
                'batch_size': 32,
                'early_stopping_patience': nn_early_stopping_patience,
                'early_stopping_min_delta': 0.0001
            }
            caar_gaar_specific_params = {'latent_dim': 64}
            mlp_pinball_params = {'quantile': 0.5} # For Median Regression
            mlp_huber_params = {'delta': 1.35} # For Huber Loss, corresponds to epsilon in sklearn's HuberRegressor

            if models['CAAR'] is None or models['CAAR'].model.input_dim != input_dim: # type: ignore
                models['CAAR'] = CAARModel(**nn_model_params, **caar_gaar_specific_params) # type: ignore
            if models['MLP'] is None or models['MLP'].model.input_dim != input_dim: # type: ignore
                models['MLP'] = MLPModel(**nn_model_params) # type: ignore
            if models['GAAR'] is None or models['GAAR'].model.input_dim != input_dim: # type: ignore
                models['GAAR'] = GAARModel(**nn_model_params, **caar_gaar_specific_params) # type: ignore
            if models['MLP_Pinball_Median'] is None or models['MLP_Pinball_Median'].model.input_dim != input_dim: # type: ignore
                models['MLP_Pinball_Median'] = MLPPinballModel(**nn_model_params, **mlp_pinball_params) # type: ignore
            if models['MLP_Huber'] is None or not hasattr(models['MLP_Huber'], 'model') or models['MLP_Huber'].model.input_dim != input_dim: # type: ignore
                models['MLP_Huber'] = MLPHuberModel(**nn_model_params, **mlp_huber_params) # type: ignore
            
            for model_name, model in models.items():
                print(f"    训练模型: {model_name}")
                
                start_time = time.time()
                model.fit(X_train, y_train, X_val, y_val, verbose=0)
                train_time = time.time() - start_time
                
                y_pred = model.predict(X_test)
                
                metrics = evaluate_model(y_test, y_pred)
                
                for metric in results['metrics']:
                    results['models'][model_name][metric].append(metrics[metric])
                
                results['train_times'][model_name].append(train_time)
                
                for metric in results['metrics']:
                    if metric not in results['trend_data'][ratio][model_name]:
                        results['trend_data'][ratio][model_name][metric] = []
                    results['trend_data'][ratio][model_name][metric].append(metrics[metric])
                
                if repeat == 0 and ratio in [outlier_ratios[0], outlier_ratios[len(outlier_ratios)//2], outlier_ratios[-1]]: # 只对部分情况绘图以节省时间
                    plot_title_suffix = f"({dataset_name.capitalize()}, Outliers: {outlier_type.upper()}, Ratio: {ratio}, Strength: {outlier_strength}, Method: {y_outlier_method if outlier_type=='y' else 'N/A'}, Seed: {current_seed})"
                    plot_prediction_vs_true(
                        y_test, y_pred, 
                        f"{model_name} {plot_title_suffix}", # 英文标题
                        save_path=os.path.join(experiment_dir, f"{model_name}_{dataset_name}_ratio{ratio}_pred_vs_true.png"),
                        show_plot=False
                    )
                    plot_residuals(
                        y_test, y_pred, 
                        f"{model_name} {plot_title_suffix}", # 英文标题
                        save_path=os.path.join(experiment_dir, f"{model_name}_{dataset_name}_ratio{ratio}_residuals.png"),
                        show_plot=False
                    )
    
    # 计算平均值和标准差
    for model_name in models.keys():
        for metric in results['metrics']:
            values = results['models'][model_name][metric]
            results['models'][model_name][f"{metric}_mean"] = np.mean(values)
            results['models'][model_name][f"{metric}_std"] = np.std(values)
        
        train_times = results['train_times'][model_name]
        results['train_times'][model_name] = {
            'mean': np.mean(train_times),
            'std': np.std(train_times)
        }
    
    trend_mean_data = {}
    for ratio in outlier_ratios:
        trend_mean_data[ratio] = {}
        for model_name in models.keys():
            trend_mean_data[ratio][model_name] = {}
            for metric in results['metrics']:
                values = results['trend_data'][ratio][model_name][metric]
                trend_mean_data[ratio][model_name][metric] = np.mean(values)
    
    results['trend_mean_data'] = trend_mean_data
    
    model_results_summary = {}
    for model_name in models.keys():
        model_results_summary[model_name] = {
            f"{m_key}": results['models'][model_name][f"{m_key}_mean"] # 使用不带 _mean 的 metric key
            for m_key in results['metrics']
        }
        model_results_summary[model_name]['Training Time (s)'] = results['train_times'][model_name]['mean'] # 英文键

    results_df = create_performance_table(model_results_summary)

    # 保存结果表格为Markdown
    markdown_table = format_performance_table_markdown(results_df)
    table_title = f"## Performance Table: {dataset_name.capitalize()} Data - Outlier Type: {outlier_type.upper()}\n(Outlier Strength: {outlier_strength}, Y-Outlier Method: {y_outlier_method if outlier_type=='y' else 'N/A'})\n\n" # 英文标题
    with open(os.path.join(experiment_dir, 'performance_table.md'), 'w') as f:
        f.write(table_title)
        f.write(markdown_table)

    # 绘制性能对比图
    for metric_key in results['metrics']: # 使用原始 metric_key
        plot_performance_comparison(
            results_df,
            metric=metric_key, # DataFrame 中的列名现在不含 _mean
            title_prefix=f"{dataset_name.capitalize()} (Outliers: {outlier_type.upper()}, Strength: {outlier_strength}, Method: {y_outlier_method if outlier_type=='y' else 'N/A'}): ", # 英文标题
            save_path=os.path.join(experiment_dir, f"performance_comparison_{metric_key}.png"),
            show_plot=False
        )

    # 绘制不同异常值比例下的性能趋势图
    for metric_key in results['metrics']:
        trend_df_data = []
        for ratio_val in outlier_ratios:
            row = {'Outlier Ratio': ratio_val} # 英文列名
            for model_name in models.keys():
                row[model_name] = trend_mean_data[ratio_val][model_name][metric_key]
            trend_df_data.append(row)
        trend_df = pd.DataFrame(trend_df_data)

        plot_trend_with_outlier_ratio(
            trend_df,
            metric=metric_key,
            title_prefix=f"{dataset_name.capitalize()} (Outliers: {outlier_type.upper()}, Strength: {outlier_strength}, Method: {y_outlier_method if outlier_type=='y' else 'N/A'}): ", # 英文标题
            save_path=os.path.join(experiment_dir, f"trend_{metric_key}.png"),
            show_plot=False
        )

    with open(os.path.join(experiment_dir, 'full_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"{dataset_name.capitalize()} 数据集实验 (异常类型: {outlier_type}, 异常强度: {outlier_strength}) 完成！") # 控制台输出可保留中文
    
    return results

def run_all_real_data_experiments(
    outlier_ratios=[0.0], # [0.0, 0.05, 0.1, 0.2], 
    n_repeats=1,
    random_state=42,
    results_dir=DEFAULT_RESULTS_DIR,
    nn_early_stopping_patience=10,
    y_outlier_method='additive_std',
    outlier_strength=5.0 # 新增参数
):
    """
    执行所有真实数据实验
    
    参数:
        outlier_ratios: 异常值比例列表
        n_repeats: 重复次数
        random_state: 随机种子
        results_dir: 结果保存目录
        nn_early_stopping_patience: 神经网络模型早停的耐心轮数
        y_outlier_method: 当outlier_type='y'时，注入y异常值的方法。
        outlier_strength: 异常值强度。
        
    返回:
        all_results: 所有实验结果字典
    """
    os.makedirs(results_dir, exist_ok=True)
    
    california_y_results = run_real_data_experiment(
        dataset_name='california',
        outlier_ratios=outlier_ratios,
        outlier_type='y',
        n_repeats=n_repeats,
        random_state=random_state,
        results_dir=results_dir,
        nn_early_stopping_patience=nn_early_stopping_patience,
        y_outlier_method=y_outlier_method,
        outlier_strength=outlier_strength # 传递参数
    )
    
    california_x_results = run_real_data_experiment(
        dataset_name='california',
        outlier_ratios=outlier_ratios,
        outlier_type='x',
        n_repeats=n_repeats,
        random_state=random_state,
        results_dir=results_dir,
        nn_early_stopping_patience=nn_early_stopping_patience,
        outlier_strength=outlier_strength # x轴异常也需要强度参数
    )
    
    diabetes_y_results = run_real_data_experiment(
        dataset_name='diabetes',
        outlier_ratios=outlier_ratios,
        outlier_type='y',
        n_repeats=n_repeats,
        random_state=random_state,
        results_dir=results_dir,
        nn_early_stopping_patience=nn_early_stopping_patience,
        y_outlier_method=y_outlier_method,
        outlier_strength=outlier_strength # 传递参数
    )

    # 新增数据集的Y轴异常实验
    new_datasets_y_experiments_results = {}
    new_dataset_names_for_y_outliers = [
        'bike_sharing',
        'parkinsons_telemonitoring',
        'boston_housing', 
        'communities_crime', 
        'concrete_strength'
    ]

    for dataset_name_iter in new_dataset_names_for_y_outliers:
        print(f"\nStarting Y-axis outlier experiments for {dataset_name_iter}...")
        exp_results = run_real_data_experiment(
            dataset_name=dataset_name_iter,
            outlier_ratios=outlier_ratios,
            outlier_type='y', # 仅 Y 轴异常
            n_repeats=n_repeats,
            random_state=random_state,
            results_dir=results_dir,
            nn_early_stopping_patience=nn_early_stopping_patience,
            y_outlier_method=y_outlier_method,
            outlier_strength=outlier_strength # 传递参数
        )
        # Sanitize key name by removing underscores for consistency if desired, or keep them
        key_name = f"{dataset_name_iter.replace('_', '')}_y"
        new_datasets_y_experiments_results[key_name] = exp_results
    
    all_results = {
        'california_y': california_y_results,
        'california_x': california_x_results,
        'diabetes_y': diabetes_y_results
    }
    all_results.update(new_datasets_y_experiments_results) # 合并新实验结果
    
    with open(os.path.join(results_dir, 'all_real_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    return all_results

if __name__ == "__main__":
    run_all_real_data_experiments(
    outlier_ratios=[0.0, 0.1, 0.2], # [0.0, 0.05, 0.1, 0.2], 
    n_repeats=3,
    random_state=42,
    results_dir=DEFAULT_RESULTS_DIR,
    nn_early_stopping_patience=10,
    y_outlier_method='sequential_multiplicative_additive',
    outlier_strength=10.0 # 示例：在主调用中指定新的强度
)


# python src/experiments/real_exp.py
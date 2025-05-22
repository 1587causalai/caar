"""
合成数据实验执行模块

执行以下实验：
1. 线性关系 + 不同比例Y异常值
2. 线性关系 + 不同比例X异常值
3. 非线性关系 + 不同比例Y异常值
4. 非线性关系 + 不同比例X异常值
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

from models.caar import CAARModel, MLPModel, GAARModel, MLPPinballModel, MLPHuberModel, MLPCauchyModel
from models.baseline import (
    OLSRegressor, 
    RandomForestRegressorWrapper,
    XGBoostRegressorWrapper
)
from data.synthetic import prepare_synthetic_experiment
from utils.metrics import evaluate_model
from utils.visualization import (
    plot_performance_comparison, 
    plot_trend_with_outlier_ratio, 
    plot_residuals, 
    plot_prediction_vs_true,
    create_performance_table,
    format_performance_table_markdown
)

def run_synthetic_linear_y_outliers_experiment(
    outlier_ratios=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
    outlier_strength=5.0,
    y_outlier_method='additive_std',
    n_samples=1000,
    n_features=10,
    n_repeats=5,
    random_state=42,
    results_dir=DEFAULT_RESULTS_DIR,
    nn_early_stopping_patience=10,
    noise_level_linear=0.5
):
    """
    执行线性关系 + 不同比例Y异常值实验
    
    参数:
        outlier_ratios: 异常值比例列表
        outlier_strength: 异常值强度
        y_outlier_method: Y轴异常值注入方法
        n_samples: 样本数量
        n_features: 特征数量
        n_repeats: 重复次数
        random_state: 随机种子
        results_dir: 结果保存目录
        nn_early_stopping_patience: 神经网络模型早停的耐心轮数
        noise_level_linear: 线性关系的基础噪声水平
        
    返回:
        results: 实验结果字典
    """
    print("开始执行线性关系 + 不同比例Y异常值实验...")
    
    experiment_name_suffix = 'synthetic_linear_y_outliers'
    experiment_dir = os.path.join(results_dir, experiment_name_suffix)
    os.makedirs(experiment_dir, exist_ok=True)
    
    results = {
        'outlier_ratios': outlier_ratios,
        'models': {},
        'metrics': ['MSE', 'RMSE', 'MAE', 'MdAE', 'R²'],
        'train_times': {},
        'trend_data': {}
    }
    
    # 定义神经网络和CAAR/GAAR特定参数，n_features即input_dim
    nn_model_params = {
        'input_dim': n_features, 
        'hidden_dims': [128, 64], 
        'epochs': 100, 
        'lr': 0.001, 
        'batch_size': 32,
        'early_stopping_patience': nn_early_stopping_patience,
        'early_stopping_min_delta': 0.0001
    }
    caar_gaar_specific_params = {'latent_dim': 64}
    mlp_pinball_params = {'quantile': 0.5}
    mlp_huber_params = {'delta': 1.35}

    models = {
        'OLS': OLSRegressor(),
        'MLP_Huber': MLPHuberModel(**nn_model_params, **mlp_huber_params),
        'RandomForest': RandomForestRegressorWrapper(n_estimators=100, random_state=random_state),
        'XGBoost': XGBoostRegressorWrapper(random_state=random_state),
        'CAAR': CAARModel(**nn_model_params, **caar_gaar_specific_params),
        'MLP': MLPModel(**nn_model_params),
        'GAAR': GAARModel(**nn_model_params, **caar_gaar_specific_params),
        'MLP_Pinball_Median': MLPPinballModel(**nn_model_params, **mlp_pinball_params),
        'MLP_Cauchy': MLPCauchyModel(**nn_model_params)
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
        
        for repeat in range(n_repeats):
            print(f"  重复 {repeat+1}/{n_repeats}")
            
            current_seed = random_state + repeat
            X_train, X_val, X_test, y_train, y_val, y_test, outlier_mask_train, outlier_mask_val = prepare_synthetic_experiment(
                n_samples_total=n_samples,
                n_features=n_features,
                outlier_ratio=ratio,
                outlier_strength=outlier_strength,
                outlier_type='y',
                relation_type='linear',
                y_outlier_method=y_outlier_method,
                random_state=current_seed,
                noise_level_linear=noise_level_linear
            )
            
            for model_name, model_instance in models.items():
                print(f"    训练模型: {model_name}")
                
                start_time = time.time()
                model_instance.fit(X_train, y_train, X_val, y_val, verbose=0)
                train_time = time.time() - start_time
                
                y_pred = model_instance.predict(X_test)
                metrics_eval = evaluate_model(y_test, y_pred)
                
                for metric_key in results['metrics']:
                    results['models'][model_name][metric_key].append(metrics_eval[metric_key])
                results['train_times'][model_name].append(train_time)
                
                for metric_key in results['metrics']:
                    if metric_key not in results['trend_data'][ratio][model_name]:
                        results['trend_data'][ratio][model_name][metric_key] = []
                    results['trend_data'][ratio][model_name][metric_key].append(metrics_eval[metric_key])
                
                if repeat == 0 and ratio in [0.0, 0.1, 0.3]:
                    plot_prediction_vs_true(
                        y_test, y_pred, 
                        f"{model_name} (Linear Y-Outliers, Ratio: {ratio}, Seed: {current_seed})",
                        save_path=os.path.join(experiment_dir, f"{model_name}_ratio{ratio}_seed{current_seed}_pred_vs_true.png"),
                        show_plot=False
                    )
                    plot_residuals(
                        y_test, y_pred, 
                        f"{model_name} (Linear Y-Outliers, Ratio: {ratio}, Seed: {current_seed})",
                        save_path=os.path.join(experiment_dir, f"{model_name}_ratio{ratio}_seed{current_seed}_residuals.png"),
                        show_plot=False
                    )
    
    # 计算平均值和标准差
    for model_name in models.keys():
        for metric_key in results['metrics']:
            values = results['models'][model_name][metric_key]
            results['models'][model_name][f"{metric_key}_mean"] = np.mean(values)
            results['models'][model_name][f"{metric_key}_std"] = np.std(values)
        
        # 计算平均训练时间
        train_times_list = results['train_times'][model_name]
        results['train_times'][model_name] = {
            'mean': np.mean(train_times_list),
            'std': np.std(train_times_list)
        }
    
    # 计算趋势数据的平均值和标准差
    trend_mean_data = {}
    for ratio in outlier_ratios:
        trend_mean_data[ratio] = {}
        for model_name in models.keys():
            trend_mean_data[ratio][model_name] = {}
            for metric_key in results['metrics']:
                values = results['trend_data'][ratio][model_name][metric_key]
                trend_mean_data[ratio][model_name][metric_key] = np.mean(values)
    
    results['trend_mean_data'] = trend_mean_data
    
    # 创建性能对比表格
    model_results_summary = {}
    for model_name in models.keys():
        model_results_summary[model_name] = {
            f"{m_key}": results['models'][model_name][f"{m_key}_mean"]
            for m_key in results['metrics']
        }
        model_results_summary[model_name]['Training Time (s)'] = results['train_times'][model_name]['mean']
    
    results_df = create_performance_table(model_results_summary)
    
    # 保存结果表格为Markdown
    markdown_table = format_performance_table_markdown(results_df)
    table_title = f"## Performance Table: Synthetic Data - Linear Relation with Y-Outliers (Strength: {outlier_strength}, Method: {y_outlier_method})\n\n"
    with open(os.path.join(experiment_dir, 'performance_table.md'), 'w') as f:
        f.write(table_title)
        f.write(markdown_table)
    
    # 绘制性能对比图
    for metric_key in results['metrics']:
        plot_performance_comparison(
            results_df, 
            metric=metric_key,
            title_prefix=f"Synthetic Linear Y-Outliers (Strength: {outlier_strength}, Method: {y_outlier_method}): ",
            save_path=os.path.join(experiment_dir, f"performance_comparison_{metric_key}.png"),
            show_plot=False
        )
    
    # 绘制不同异常值比例下的性能趋势图
    for metric_key in results['metrics']:
        # 准备趋势数据
        trend_df_data = []
        for ratio_val in outlier_ratios:
            row = {'Outlier Ratio': ratio_val}
            for model_name in models.keys():
                row[model_name] = trend_mean_data[ratio_val][model_name][metric_key]
            trend_df_data.append(row)
        trend_df = pd.DataFrame(trend_df_data)
        
        plot_trend_with_outlier_ratio(
            trend_df,
            metric=metric_key,
            title_prefix=f"Synthetic Linear Y-Outliers (Strength: {outlier_strength}, Method: {y_outlier_method}): ",
            save_path=os.path.join(experiment_dir, f"trend_{metric_key}.png"),
            show_plot=False
        )
    
    # 保存完整结果
    with open(os.path.join(experiment_dir, 'full_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("线性关系 + 不同比例Y异常值实验完成！")
    
    return results

def run_synthetic_linear_x_outliers_experiment(
    outlier_ratios=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
    outlier_strength=5.0,
    n_samples=1000,
    n_features=10,
    n_repeats=5,
    random_state=42,
    results_dir=DEFAULT_RESULTS_DIR,
    nn_early_stopping_patience=10,
    noise_level_linear=0.5
):
    """
    执行线性关系 + 不同比例X异常值实验
    
    参数:
        outlier_ratios: 异常值比例列表
        outlier_strength: 异常值强度
        n_samples: 样本数量
        n_features: 特征数量
        n_repeats: 重复次数
        random_state: 随机种子
        results_dir: 结果保存目录
        nn_early_stopping_patience: 神经网络模型早停的耐心轮数
        noise_level_linear: 线性关系的基础噪声水平
        
    返回:
        results: 实验结果字典
    """
    print("开始执行线性关系 + 不同比例X异常值实验...")
    
    experiment_name_suffix = 'synthetic_linear_x_outliers'
    experiment_dir = os.path.join(results_dir, experiment_name_suffix)
    experiment_dir = os.path.join(results_dir, 'synthetic_linear_x_outliers')
    os.makedirs(experiment_dir, exist_ok=True)
    
    results = {
        'outlier_ratios': outlier_ratios,
        'models': {},
        'metrics': ['MSE', 'RMSE', 'MAE', 'MdAE', 'R²'],
        'train_times': {},
        'trend_data': {}
    }
    
    # 定义神经网络和CAAR/GAAR特定参数，n_features即input_dim
    nn_model_params = {
        'input_dim': n_features, 
        'hidden_dims': [128, 64], 
        'epochs': 100, 
        'lr': 0.001, 
        'batch_size': 32,
        'early_stopping_patience': nn_early_stopping_patience,
        'early_stopping_min_delta': 0.0001
    }
    caar_gaar_specific_params = {'latent_dim': 64}
    mlp_pinball_params = {'quantile': 0.5}
    mlp_huber_params = {'delta': 1.35}

    models = {
        'OLS': OLSRegressor(),
        'MLP_Huber': MLPHuberModel(**nn_model_params, **mlp_huber_params),
        'RandomForest': RandomForestRegressorWrapper(n_estimators=100, random_state=random_state),
        'XGBoost': XGBoostRegressorWrapper(random_state=random_state),
        'CAAR': CAARModel(**nn_model_params, **caar_gaar_specific_params),
        'MLP': MLPModel(**nn_model_params),
        'GAAR': GAARModel(**nn_model_params, **caar_gaar_specific_params),
        'MLP_Pinball_Median': MLPPinballModel(**nn_model_params, **mlp_pinball_params),
        'MLP_Cauchy': MLPCauchyModel(**nn_model_params)
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
        
        for repeat in range(n_repeats):
            print(f"  重复 {repeat+1}/{n_repeats}")
            
            current_seed = random_state + repeat
            X_train, X_val, X_test, y_train, y_val, y_test, outlier_mask_train, outlier_mask_val = prepare_synthetic_experiment(
                n_samples_total=n_samples,
                n_features=n_features,
                outlier_ratio=ratio,
                outlier_strength=outlier_strength,
                outlier_type='x',
                relation_type='linear',
                random_state=current_seed,
                noise_level_linear=noise_level_linear
            )
            
            for model_name, model_instance in models.items():
                print(f"    训练模型: {model_name}")
                
                start_time = time.time()
                model_instance.fit(X_train, y_train, X_val, y_val, verbose=0)
                train_time = time.time() - start_time
                
                y_pred = model_instance.predict(X_test)
                metrics_eval = evaluate_model(y_test, y_pred)
                
                for metric_key in results['metrics']:
                    results['models'][model_name][metric_key].append(metrics_eval[metric_key])
                results['train_times'][model_name].append(train_time)
                
                for metric_key in results['metrics']:
                    if metric_key not in results['trend_data'][ratio][model_name]:
                        results['trend_data'][ratio][model_name][metric_key] = []
                    results['trend_data'][ratio][model_name][metric_key].append(metrics_eval[metric_key])
                
                if repeat == 0 and ratio in [0.0, 0.1, 0.3]:
                    plot_prediction_vs_true(
                        y_test, y_pred, 
                        f"{model_name} (Linear X-Outliers, Ratio: {ratio}, Seed: {current_seed})",
                        save_path=os.path.join(experiment_dir, f"{model_name}_ratio{ratio}_seed{current_seed}_pred_vs_true.png"),
                        show_plot=False
                    )
                    plot_residuals(
                        y_test, y_pred, 
                        f"{model_name} (Linear X-Outliers, Ratio: {ratio}, Seed: {current_seed})",
                        save_path=os.path.join(experiment_dir, f"{model_name}_ratio{ratio}_seed{current_seed}_residuals.png"),
                        show_plot=False
                    )
    
    # 计算平均值和标准差
    for model_name in models.keys():
        for metric_key in results['metrics']:
            values = results['models'][model_name][metric_key]
            results['models'][model_name][f"{metric_key}_mean"] = np.mean(values)
            results['models'][model_name][f"{metric_key}_std"] = np.std(values)
        
        # 计算平均训练时间
        train_times_list = results['train_times'][model_name]
        results['train_times'][model_name] = {
            'mean': np.mean(train_times_list),
            'std': np.std(train_times_list)
        }
    
    # 计算趋势数据的平均值和标准差
    trend_mean_data = {}
    for ratio in outlier_ratios:
        trend_mean_data[ratio] = {}
        for model_name in models.keys():
            trend_mean_data[ratio][model_name] = {}
            for metric_key in results['metrics']:
                values = results['trend_data'][ratio][model_name][metric_key]
                trend_mean_data[ratio][model_name][metric_key] = np.mean(values)
    
    results['trend_mean_data'] = trend_mean_data
    
    # 创建性能对比表格
    model_results_summary = {}
    for model_name in models.keys():
        model_results_summary[model_name] = {
            f"{m_key}": results['models'][model_name][f"{m_key}_mean"]
            for m_key in results['metrics']
        }
        model_results_summary[model_name]['Training Time (s)'] = results['train_times'][model_name]['mean']
    
    results_df = create_performance_table(model_results_summary)
    
    # 保存结果表格为Markdown
    markdown_table = format_performance_table_markdown(results_df)
    table_title = f"## Performance Table: Synthetic Data - Linear Relation with X-Outliers (Strength: {outlier_strength})\n\n"
    with open(os.path.join(experiment_dir, 'performance_table.md'), 'w') as f:
        f.write(table_title)
        f.write(markdown_table)
    
    # 绘制性能对比图
    for metric_key in results['metrics']:
        plot_performance_comparison(
            results_df, 
            metric=metric_key,
            title_prefix=f"Synthetic Linear X-Outliers (Strength: {outlier_strength}): ",
            save_path=os.path.join(experiment_dir, f"performance_comparison_{metric_key}.png"),
            show_plot=False
        )
    
    # 绘制不同异常值比例下的性能趋势图
    for metric_key in results['metrics']:
        # 准备趋势数据
        trend_df_data = []
        for ratio_val in outlier_ratios:
            row = {'Outlier Ratio': ratio_val}
            for model_name in models.keys():
                row[model_name] = trend_mean_data[ratio_val][model_name][metric_key]
            trend_df_data.append(row)
        trend_df = pd.DataFrame(trend_df_data)
        
        plot_trend_with_outlier_ratio(
            trend_df,
            metric=metric_key,
            title_prefix=f"Synthetic Linear X-Outliers (Strength: {outlier_strength}): ",
            save_path=os.path.join(experiment_dir, f"trend_{metric_key}.png"),
            show_plot=False
        )
    
    # 保存完整结果
    with open(os.path.join(experiment_dir, 'full_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("线性关系 + 不同比例X异常值实验完成！")
    
    return results

def run_synthetic_nonlinear_y_outliers_experiment(
    outlier_ratios=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
    outlier_strength=5.0,
    y_outlier_method='additive_std',
    relation_type='polynomial',
    n_samples=1000,
    n_features=10,
    n_repeats=5,
    random_state=42,
    results_dir=DEFAULT_RESULTS_DIR,
    nn_early_stopping_patience=10,
    noise_level_nonlinear=0.5,
    n_interaction_terms=3,
    interaction_strength=0.5,
    hetero_strength=0.5,
    main_effect_strength=1.0
):
    """
    执行非线性关系 + 不同比例Y异常值实验
    
    参数:
        outlier_ratios: 异常值比例列表
        outlier_strength: 异常值强度
        y_outlier_method: Y轴异常值注入方法
        relation_type: 非线性关系类型 ('polynomial', 'sine', 'exp', 'interactive_heteroscedastic')
        n_samples: 样本数量
        n_features: 特征数量
        n_repeats: 重复次数
        random_state: 随机种子
        results_dir: 结果保存目录
        nn_early_stopping_patience: 神经网络模型早停的耐心轮数
        noise_level_nonlinear: 非线性关系的基础噪声水平
        n_interaction_terms: (for 'interactive_heteroscedastic')
        interaction_strength: (for 'interactive_heteroscedastic')
        hetero_strength: (for 'interactive_heteroscedastic')
        main_effect_strength: (for 'interactive_heteroscedastic')
        
    返回:
        results: 实验结果字典
    """
    print("开始执行非线性关系 + 不同比例Y异常值实验...")
    
    experiment_dir = os.path.join(results_dir, 'synthetic_nonlinear_y_outliers')
    os.makedirs(experiment_dir, exist_ok=True)
    
    results = {
        'outlier_ratios': outlier_ratios,
        'models': {},
        'metrics': ['MSE', 'RMSE', 'MAE', 'MdAE', 'R²'],
        'train_times': {},
        'trend_data': {}
    }
    
    # 定义神经网络和CAAR/GAAR特定参数，n_features即input_dim
    nn_model_params_nonlinear = {
        'input_dim': n_features, 
        'hidden_dims': [256, 128],
        'epochs': 150,
        'lr': 0.001, 
        'batch_size': 32,
        'early_stopping_patience': nn_early_stopping_patience,
        'early_stopping_min_delta': 0.0001
    }
    caar_gaar_specific_params_nonlinear = {'latent_dim': 128}
    mlp_pinball_params_nonlinear = {'quantile': 0.5}
    mlp_huber_params_nonlinear = {'delta': 1.35}

    models = {
        'RandomForest': RandomForestRegressorWrapper(n_estimators=100, max_depth=None, random_state=random_state),
        'MLP_Huber': MLPHuberModel(**nn_model_params_nonlinear, **mlp_huber_params_nonlinear),
        'XGBoost': XGBoostRegressorWrapper(random_state=random_state),
        'CAAR': CAARModel(**nn_model_params_nonlinear, **caar_gaar_specific_params_nonlinear),
        'MLP': MLPModel(**nn_model_params_nonlinear),
        'GAAR': GAARModel(**nn_model_params_nonlinear, **caar_gaar_specific_params_nonlinear),
        'MLP_Pinball_Median': MLPPinballModel(**nn_model_params_nonlinear, **mlp_pinball_params_nonlinear),
        'MLP_Cauchy': MLPCauchyModel(**nn_model_params_nonlinear)
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
        
        for repeat in range(n_repeats):
            print(f"  重复 {repeat+1}/{n_repeats}")
            
            current_seed = random_state + repeat
            X_train, X_val, X_test, y_train, y_val, y_test, outlier_mask_train, outlier_mask_val = prepare_synthetic_experiment(
                n_samples_total=n_samples,
                n_features=n_features,
                outlier_ratio=ratio,
                outlier_strength=outlier_strength,
                outlier_type='y',
                relation_type=relation_type,
                y_outlier_method=y_outlier_method,
                random_state=current_seed,
                noise_level_nonlinear=noise_level_nonlinear,
                n_interaction_terms=n_interaction_terms,
                interaction_strength=interaction_strength,
                hetero_strength=hetero_strength,
                main_effect_strength=main_effect_strength
            )
            
            for model_name, model_instance in models.items():
                print(f"    训练模型: {model_name}")
                
                start_time = time.time()
                model_instance.fit(X_train, y_train, X_val, y_val, verbose=0)
                train_time = time.time() - start_time
                
                y_pred = model_instance.predict(X_test)
                metrics_eval = evaluate_model(y_test, y_pred)
                
                for metric_key in results['metrics']:
                    results['models'][model_name][metric_key].append(metrics_eval[metric_key])
                results['train_times'][model_name].append(train_time)
                
                for metric_key in results['metrics']:
                    if metric_key not in results['trend_data'][ratio][model_name]:
                        results['trend_data'][ratio][model_name][metric_key] = []
                    results['trend_data'][ratio][model_name][metric_key].append(metrics_eval[metric_key])
                
                if repeat == 0 and ratio in [0.0, 0.1, 0.3]:
                    plot_prediction_vs_true(
                        y_test, y_pred, 
                        f"{model_name} (Nonlinear Y-Outliers, Type: {relation_type}, Ratio: {ratio}, Seed: {current_seed})",
                        save_path=os.path.join(experiment_dir, f"{model_name}_type{relation_type}_ratio{ratio}_seed{current_seed}_pred_vs_true.png"),
                        show_plot=False
                    )
                    plot_residuals(
                        y_test, y_pred, 
                        f"{model_name} (Nonlinear Y-Outliers, Type: {relation_type}, Ratio: {ratio}, Seed: {current_seed})",
                        save_path=os.path.join(experiment_dir, f"{model_name}_type{relation_type}_ratio{ratio}_seed{current_seed}_residuals.png"),
                        show_plot=False
                    )
    
    # 计算平均值和标准差
    for model_name in models.keys():
        for metric_key in results['metrics']:
            values = results['models'][model_name][metric_key]
            results['models'][model_name][f"{metric_key}_mean"] = np.mean(values)
            results['models'][model_name][f"{metric_key}_std"] = np.std(values)
        
        # 计算平均训练时间
        train_times_list = results['train_times'][model_name]
        results['train_times'][model_name] = {
            'mean': np.mean(train_times_list),
            'std': np.std(train_times_list)
        }
    
    # 计算趋势数据的平均值和标准差
    trend_mean_data = {}
    for ratio in outlier_ratios:
        trend_mean_data[ratio] = {}
        for model_name in models.keys():
            trend_mean_data[ratio][model_name] = {}
            for metric_key in results['metrics']:
                values = results['trend_data'][ratio][model_name][metric_key]
                trend_mean_data[ratio][model_name][metric_key] = np.mean(values)
    
    results['trend_mean_data'] = trend_mean_data
    
    # 创建性能对比表格
    model_results_summary = {}
    for model_name in models.keys():
        model_results_summary[model_name] = {
            f"{m_key}": results['models'][model_name][f"{m_key}_mean"]
            for m_key in results['metrics']
        }
        model_results_summary[model_name]['Training Time (s)'] = results['train_times'][model_name]['mean']
    
    results_df = create_performance_table(model_results_summary)
    
    # 保存结果表格为Markdown
    markdown_table = format_performance_table_markdown(results_df)
    table_title = f"## Performance Table: Synthetic Data - Nonlinear Relation ({relation_type}) with Y-Outliers (Strength: {outlier_strength}, Method: {y_outlier_method})\n\n"
    with open(os.path.join(experiment_dir, 'performance_table.md'), 'w') as f:
        f.write(table_title)
        f.write(markdown_table)
    
    # 绘制性能对比图
    for metric_key in results['metrics']:
        plot_performance_comparison(
            results_df, 
            metric=metric_key,
            title_prefix=f"Synthetic Nonlinear Y-Outliers ({relation_type}, Strength: {outlier_strength}, Method: {y_outlier_method}): ",
            save_path=os.path.join(experiment_dir, f"performance_comparison_{metric_key}.png"),
            show_plot=False
        )
    
    # 绘制不同异常值比例下的性能趋势图
    for metric_key in results['metrics']:
        # 准备趋势数据
        trend_df_data = []
        for ratio_val in outlier_ratios:
            row = {'Outlier Ratio': ratio_val}
            for model_name in models.keys():
                row[model_name] = trend_mean_data[ratio_val][model_name][metric_key]
            trend_df_data.append(row)
        trend_df = pd.DataFrame(trend_df_data)
        
        plot_trend_with_outlier_ratio(
            trend_df,
            metric=metric_key,
            title_prefix=f"Synthetic Nonlinear Y-Outliers ({relation_type}, Strength: {outlier_strength}, Method: {y_outlier_method}): ",
            save_path=os.path.join(experiment_dir, f"trend_{metric_key}.png"),
            show_plot=False
        )
    
    # 保存完整结果
    with open(os.path.join(experiment_dir, 'full_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("非线性关系 + 不同比例Y异常值实验完成！")
    
    return results

def run_all_synthetic_experiments(
    outlier_ratios=[0.0, 0.05, 0.1, 0.2],
    n_samples_total=500,
    n_features=5,
    n_repeats=3,
    random_state=42,
    results_dir=DEFAULT_RESULTS_DIR,
    nn_early_stopping_patience=10,
    outlier_strength=5.0,
    y_outlier_method='additive_std',
    nonlinear_relation_type='polynomial',
    noise_level_linear=0.5,
    noise_level_nonlinear=0.5
):
    """
    执行所有合成数据实验
    
    参数:
        outlier_ratios: 异常值比例列表
        n_samples_total: 样本数量
        n_features: 特征数量
        n_repeats: 重复次数
        random_state: 随机种子
        results_dir: 结果保存目录
        nn_early_stopping_patience: 神经网络模型早停的耐心轮数
        outlier_strength: 异常值强度 (用于Y轴异常)
        y_outlier_method: Y轴异常值注入方法
        nonlinear_relation_type: 用于非线性Y轴异常实验的关系类型
        noise_level_linear: 线性关系中的噪声水平
        noise_level_nonlinear: 非线性关系中的噪声水平
        
    返回:
        all_results: 所有实验结果字典
    """
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 执行线性关系 + Y异常值实验
    linear_y_results = run_synthetic_linear_y_outliers_experiment(
        outlier_ratios=outlier_ratios,
        n_samples=n_samples_total,
        n_features=n_features,
        n_repeats=n_repeats,
        random_state=random_state,
        results_dir=results_dir,
        nn_early_stopping_patience=nn_early_stopping_patience,
        outlier_strength=outlier_strength,
        y_outlier_method=y_outlier_method,
        noise_level_linear=noise_level_linear
    )
    
    # 执行线性关系 + X异常值实验
    # linear_x_results = run_synthetic_linear_x_outliers_experiment(
    #     outlier_ratios=outlier_ratios,
    #     n_samples=n_samples_total,
    #     n_features=n_features,
    #     n_repeats=n_repeats,
    #     random_state=random_state,
    #     results_dir=results_dir,
    #     nn_early_stopping_patience=nn_early_stopping_patience,
    #     outlier_strength=outlier_strength
    # )
    
    # 执行非线性关系 + Y异常值实验
    nonlinear_y_results = run_synthetic_nonlinear_y_outliers_experiment(
        outlier_ratios=outlier_ratios,
        n_samples=n_samples_total,
        n_features=n_features,
        n_repeats=n_repeats,
        random_state=random_state,
        results_dir=results_dir,
        nn_early_stopping_patience=nn_early_stopping_patience,
        outlier_strength=outlier_strength,
        y_outlier_method=y_outlier_method,
        relation_type=nonlinear_relation_type,
        noise_level_nonlinear=noise_level_nonlinear
    )
    
    # 汇总所有结果
    all_results = {
        'linear_y': linear_y_results,
        # 'linear_x': linear_x_results,
        'nonlinear_y': nonlinear_y_results
    }
    
    # 保存汇总结果
    with open(os.path.join(results_dir, 'all_synthetic_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    return all_results

if __name__ == "__main__":
    # 执行所有合成数据实验
    run_all_synthetic_experiments(
        outlier_ratios=[0.0, 0.1, 0.2],
        n_samples_total=5000,
        n_features=100,
        n_repeats=1,
        random_state=42,
        results_dir=DEFAULT_RESULTS_DIR,
        nn_early_stopping_patience=10,
        outlier_strength=5.0,
        y_outlier_method='sequential_multiplicative_additive',
        nonlinear_relation_type='interactive_heteroscedastic',
        noise_level_linear=2.0,
        noise_level_nonlinear=0.5
    )

"""
评估指标模块

提供以下功能：
1. 计算均方误差 (Mean Squared Error, MSE)
2. 计算均方根误差 (Root Mean Squared Error, RMSE)
3. 计算平均绝对误差 (Mean Absolute Error, MAE)
4. 计算中位数绝对误差 (Median Absolute Error, MdAE)
5. 计算决定系数 (R-squared, R²)
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_mse(y_true, y_pred):
    """
    计算均方误差 (Mean Squared Error, MSE)
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        mse: 均方误差
    """
    return mean_squared_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    """
    计算均方根误差 (Root Mean Squared Error, RMSE)
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        rmse: 均方根误差
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """
    计算平均绝对误差 (Mean Absolute Error, MAE)
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        mae: 平均绝对误差
    """
    return mean_absolute_error(y_true, y_pred)

def calculate_mdae(y_true, y_pred):
    """
    计算中位数绝对误差 (Median Absolute Error, MdAE)
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        mdae: 中位数绝对误差
    """
    return np.median(np.abs(y_true - y_pred))

def calculate_r2(y_true, y_pred):
    """
    计算决定系数 (R-squared, R²)
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        r2: 决定系数
    """
    return r2_score(y_true, y_pred)

def evaluate_model(y_true, y_pred):
    """
    全面评估模型性能
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        metrics: 包含所有评估指标的字典
    """
    metrics = {
        'MSE': calculate_mse(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'MdAE': calculate_mdae(y_true, y_pred),
        'R²': calculate_r2(y_true, y_pred)
    }
    
    return metrics

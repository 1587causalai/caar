"""
现有鲁棒回归方法实现

包含以下鲁棒性回归方法：
1. Huber回归 (Huber Regressor)
2. 分位数回归 (Quantile Regressor)
3. RANSAC回归 (Random Sample Consensus Regressor)
"""

import numpy as np
import time
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class HuberRegressorWrapper:
    """
    Huber回归 (Huber Regressor)
    
    使用Huber损失函数，对小误差采用平方惩罚（L2），对大误差采用线性惩罚（L1），
    有效降低了异常点的影响，是标准的鲁棒回归基线。
    """
    def __init__(self, epsilon=1.35, max_iter=100, alpha=0.0001):
        """
        初始化Huber回归器
        
        参数:
            epsilon: Huber损失函数的参数，控制L1和L2损失之间的切换点
            max_iter: 最大迭代次数
            alpha: 正则化强度
        """
        self.model = HuberRegressor(
            epsilon=epsilon,
            max_iter=max_iter,
            alpha=alpha
        )
        self.history = {'train_time': 0}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """
        训练模型
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征 (未使用，保持接口一致)
            y_val: 验证集标签 (未使用，保持接口一致)
            verbose: 是否打印训练信息
            
        返回:
            self: 训练好的模型
        """
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 记录训练结束时间
        end_time = time.time()
        self.history['train_time'] = end_time - start_time
        
        if verbose:
            print(f'Huber回归训练完成，耗时 {self.history["train_time"]:.2f} 秒')
            print(f'模型系数: {self.model.coef_}')
            print(f'模型截距: {self.model.intercept_}')
        
        return self
    
    def predict(self, X):
        """
        预测函数
        
        参数:
            X: 测试集特征
            
        返回:
            y_pred: 预测结果
        """
        return self.model.predict(X)
    
    def get_params(self):
        """
        获取模型参数
        
        返回:
            params: 模型参数字典
        """
        return {
            'epsilon': self.model.epsilon,
            'max_iter': self.model.max_iter,
            'alpha': self.model.alpha
        }
    
    def set_params(self, **params):
        """
        设置模型参数
        
        参数:
            params: 模型参数字典
            
        返回:
            self: 更新参数后的模型
        """
        if 'epsilon' in params:
            self.model.epsilon = params['epsilon']
        
        if 'max_iter' in params:
            self.model.max_iter = params['max_iter']
        
        if 'alpha' in params:
            self.model.alpha = params['alpha']
        
        return self

class QuantileRegressorWrapper:
    """
    分位数回归 (Quantile Regressor)
    
    不拟合均值，而是拟合条件分位数（例如，0.5分位数即中位数）。
    中位数天生对异常值不敏感，使其成为一种非常鲁棒的回归方法。
    """
    def __init__(self, quantile=0.5, alpha=1.0, solver='highs'):
        """
        初始化分位数回归器
        
        参数:
            quantile: 分位数，默认为0.5（中位数回归）
            alpha: 正则化强度
            solver: 求解器
        """
        self.model = QuantileRegressor(
            quantile=quantile,
            alpha=alpha,
            solver=solver
        )
        self.history = {'train_time': 0}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """
        训练模型
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征 (未使用，保持接口一致)
            y_val: 验证集标签 (未使用，保持接口一致)
            verbose: 是否打印训练信息
            
        返回:
            self: 训练好的模型
        """
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 记录训练结束时间
        end_time = time.time()
        self.history['train_time'] = end_time - start_time
        
        if verbose:
            print(f'分位数回归训练完成，耗时 {self.history["train_time"]:.2f} 秒')
            print(f'模型系数: {self.model.coef_}')
            print(f'模型截距: {self.model.intercept_}')
        
        return self
    
    def predict(self, X):
        """
        预测函数
        
        参数:
            X: 测试集特征
            
        返回:
            y_pred: 预测结果
        """
        return self.model.predict(X)
    
    def get_params(self):
        """
        获取模型参数
        
        返回:
            params: 模型参数字典
        """
        return {
            'quantile': self.model.quantile,
            'alpha': self.model.alpha,
            'solver': self.model.solver
        }
    
    def set_params(self, **params):
        """
        设置模型参数
        
        参数:
            params: 模型参数字典
            
        返回:
            self: 更新参数后的模型
        """
        if 'quantile' in params:
            self.model.quantile = params['quantile']
        
        if 'alpha' in params:
            self.model.alpha = params['alpha']
        
        if 'solver' in params:
            self.model.solver = params['solver']
        
        return self

class RANSACRegressorWrapper:
    """
    RANSAC回归 (Random Sample Consensus Regressor)
    
    通过迭代地随机选择数据子集来拟合模型，并识别内点（inliers）和外点（outliers），
    是处理异常点效果显著的方法。
    """
    def __init__(self, min_samples=None, residual_threshold=None, max_trials=100, random_state=None):
        """
        初始化RANSAC回归器
        
        参数:
            min_samples: 最小样本数
            residual_threshold: 残差阈值
            max_trials: 最大迭代次数
            random_state: 随机种子
        """
        self.model = RANSACRegressor(
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            random_state=random_state
        )
        self.history = {'train_time': 0}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        """
        训练模型
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征 (未使用，保持接口一致)
            y_val: 验证集标签 (未使用，保持接口一致)
            verbose: 是否打印训练信息
            
        返回:
            self: 训练好的模型
        """
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 记录训练结束时间
        end_time = time.time()
        self.history['train_time'] = end_time - start_time
        
        if verbose:
            print(f'RANSAC回归训练完成，耗时 {self.history["train_time"]:.2f} 秒')
            print(f'内点数量: {np.sum(self.model.inlier_mask_)}')
            print(f'外点数量: {np.sum(~self.model.inlier_mask_)}')
        
        return self
    
    def predict(self, X):
        """
        预测函数
        
        参数:
            X: 测试集特征
            
        返回:
            y_pred: 预测结果
        """
        return self.model.predict(X)
    
    def get_params(self):
        """
        获取模型参数
        
        返回:
            params: 模型参数字典
        """
        return {
            'min_samples': self.model.min_samples,
            'residual_threshold': self.model.residual_threshold,
            'max_trials': self.model.max_trials,
            'random_state': self.model.random_state
        }
    
    def set_params(self, **params):
        """
        设置模型参数
        
        参数:
            params: 模型参数字典
            
        返回:
            self: 更新参数后的模型
        """
        if 'min_samples' in params:
            self.model.min_samples = params['min_samples']
        
        if 'residual_threshold' in params:
            self.model.residual_threshold = params['residual_threshold']
        
        if 'max_trials' in params:
            self.model.max_trials = params['max_trials']
        
        if 'random_state' in params:
            self.model.random_state = params['random_state']
        
        return self

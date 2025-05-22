"""
基线回归模型实现

包含以下非鲁棒性/高性能基线方法：
1. 普通最小二乘法 (Ordinary Least Squares, OLS)
2. 岭回归 (Ridge Regression)
3. 随机森林回归 (Random Forest Regressor)
"""

import numpy as np
import time
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

class OLSRegressor:
    """
    普通最小二乘法 (Ordinary Least Squares, OLS)
    
    最基础的线性回归模型，对异常值极其敏感。
    """
    def __init__(self, fit_intercept=True):
        """
        初始化OLS回归器
        
        参数:
            fit_intercept: 是否拟合截距
        """
        self.model = LinearRegression(fit_intercept=fit_intercept)
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
            print(f'OLS训练完成，耗时 {self.history["train_time"]:.2f} 秒')
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
        return {'fit_intercept': self.model.fit_intercept}
    
    def set_params(self, **params):
        """
        设置模型参数
        
        参数:
            params: 模型参数字典
            
        返回:
            self: 更新参数后的模型
        """
        if 'fit_intercept' in params:
            self.model.fit_intercept = params['fit_intercept']
        
        return self

class RidgeRegressor:
    """
    岭回归 (Ridge Regression)
    
    通过L2正则化解决多重共线性，对模型参数有约束，但在异常点存在时，其对预测性能的改善有限。
    """
    def __init__(self, alpha=1.0, fit_intercept=True):
        """
        初始化岭回归器
        
        参数:
            alpha: 正则化强度
            fit_intercept: 是否拟合截距
        """
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
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
            print(f'岭回归训练完成，耗时 {self.history["train_time"]:.2f} 秒')
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
            'alpha': self.model.alpha,
            'fit_intercept': self.model.fit_intercept
        }
    
    def set_params(self, **params):
        """
        设置模型参数
        
        参数:
            params: 模型参数字典
            
        返回:
            self: 更新参数后的模型
        """
        if 'alpha' in params:
            self.model.alpha = params['alpha']
        
        if 'fit_intercept' in params:
            self.model.fit_intercept = params['fit_intercept']
        
        return self

class RandomForestRegressorWrapper:
    """
    随机森林回归 (Random Forest Regressor)
    
    一种强大的集成学习方法，通过多棵决策树的集成来提高预测精度和稳定性。
    虽然树模型对异常值有一定抵抗力，但并非专门的鲁棒算法。
    """
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        """
        初始化随机森林回归器
        
        参数:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            random_state: 随机种子
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
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
            print(f'随机森林回归训练完成，耗时 {self.history["train_time"]:.2f} 秒')
            print(f'特征重要性: {self.model.feature_importances_}')
        
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
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
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
        if 'n_estimators' in params:
            self.model.n_estimators = params['n_estimators']
        
        if 'max_depth' in params:
            self.model.max_depth = params['max_depth']
        
        if 'random_state' in params:
            self.model.random_state = params['random_state']
        
        return self

class XGBoostRegressorWrapper:
    """
    XGBoost Regressor Wrapper
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None, **kwargs):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        self.history = {'train_time': 0}

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        start_time = time.time()
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # XGBoost 的 fit 方法有 early_stopping_rounds 参数，但 verbose 控制台输出方式不同
        # 为了简单起见，我们不在这里显式使用 early_stopping_rounds，而是依赖固定的 n_estimators
        # 或者可以在 kwargs 中传入 early_stopping_rounds 和 eval_metric
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False if verbose == 0 else True)
        
        self.history['train_time'] = time.time() - start_time
        if verbose:
            print(f'XGBoost Regressor training completed in {self.history["train_time"]:.2f} seconds')
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

class LightGBMRegressorWrapper:
    """
    LightGBM Regressor Wrapper
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=None, **kwargs):
        # LightGBM 的 verbose 参数：-1 静默, 0 警告, 1 信息
        # 我们将通过 kwargs 传递 verbose，如果未提供则默认为 0 (警告)
        lgb_verbose = kwargs.pop('verbose', 0) 
        if lgb_verbose == 0 and 'callbacks' not in kwargs: # 实验脚本中 verbose=0 表示不打印，映射到 lgb 的 -1
            lgb_verbose = -1

        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            verbose=lgb_verbose, # 控制 LightGBM 自身的日志级别
            **kwargs
        )
        self.history = {'train_time': 0}

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=1):
        start_time = time.time()
        eval_set = []
        callbacks = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # LightGBM 的 fit 方法中，verbose 参数控制的是迭代过程的打印频率
        # 为了与我们脚本的 verbose (0 或 1) 保持行为一致 (即是否打印完成信息)
        # LightGBM 自身的日志级别已在 __init__ 中通过 verbose 参数设置
        # 如果需要早停，可以通过 kwargs 传入 early_stopping_rounds 和 eval_metric，并构造 callback
        # 例如: if 'early_stopping_rounds' in self.model.get_params():
        # callbacks.append(lgb.early_stopping(self.model.early_stopping_rounds, verbose=False))

        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        
        self.history['train_time'] = time.time() - start_time
        if verbose:
             print(f'LightGBM Regressor training completed in {self.history["train_time"]:.2f} seconds')
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

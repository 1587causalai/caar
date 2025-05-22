"""
合成数据生成模块

提供以下功能：
1. 生成具有线性关系的合成数据
2. 生成具有非线性关系的合成数据
3. 在数据中注入Y-异常值
4. 在数据中注入X-异常值
"""

import numpy as np
from sklearn.model_selection import train_test_split

class SyntheticDataGenerator:
    def __init__(self, n_samples_total=1000, n_features=10, random_state=None):
        """
        初始化合成数据生成器
        
        参数:
            n_samples_total: 生成的总样本数量，后续会进行划分
            n_features: 特征数量
            random_state: 随机种子
        """
        self.n_samples_total = n_samples_total
        self.n_features = n_features
        self.rng = np.random.RandomState(random_state)
        self.X = None  # 用于存储生成的特征
        self.y = None  # 用于存储生成的目标变量
        self.true_coef_ = None # 仅用于线性情况

    def generate_linear(self, noise_level=0.5):
        """
        生成具有线性关系的合成数据
        
        参数:
            noise_level: 噪声水平
            
        返回:
            X: 特征矩阵，形状为 (self.n_samples_total, self.n_features)
            y: 目标变量，形状为 (self.n_samples_total,)
        """
        self.X = self.rng.randn(self.n_samples_total, self.n_features)
        self.true_coef_ = self.rng.randn(self.n_features)
        self.y = np.dot(self.X, self.true_coef_) + noise_level * self.rng.randn(self.n_samples_total)
        return self.X, self.y

    def generate_nonlinear(self, function_type='polynomial', noise_level=0.5, 
                            n_interaction_terms=10, interaction_strength=0.5,
                            hetero_strength=0.5, main_effect_strength=1.0):
        """
        生成具有非线性关系的合成数据
        
        参数:
            function_type: 非线性函数类型
            noise_level: 基础噪声水平
            n_interaction_terms: 交叉项数量
            interaction_strength: 交叉项强度
            hetero_strength: 异方差噪声强度
            main_effect_strength: 主要非线性效应强度
            
        返回:
            X: 特征矩阵
            y: 目标变量
        """
        self.X = self.rng.randn(self.n_samples_total, self.n_features)
        y_signal_components = []

        if function_type == 'polynomial':
            self.y = np.sum(self.X**2, axis=1) + noise_level * self.rng.randn(self.n_samples_total)
        
        elif function_type == 'sine':
            self.y = np.sin(np.sum(self.X, axis=1)) + noise_level * self.rng.randn(self.n_samples_total)
        
        elif function_type == 'exp':
            self.y = np.exp(np.sum(self.X * 0.1, axis=1)) + noise_level * self.rng.randn(self.n_samples_total)

        elif function_type == 'interactive_heteroscedastic':
            if self.n_features < 2 and n_interaction_terms > 0:
                raise ValueError("Cannot create interaction terms with less than 2 features.")
            
            # --- 构建共享的非线性基础函数库 ---
            basis_functions = []
            # basis_function_types = [] # 用于调试或理解生成器 (保留注释以备将来使用)
            
            # 1. 线性主效应
            for i in range(self.n_features):
                basis_functions.append(self.X[:, i])
                # basis_function_types.append(f'X_{i}_linear')

            # 2. 非线性主效应
            for i in range(self.n_features):
                # 多项式项
                basis_functions.append(self.X[:, i]**2)
                # basis_function_types.append(f'X_{i}_poly2')
                # 正弦项
                basis_functions.append(np.sin(self.X[:, i] * self.rng.uniform(0.5, 1.5))) # 随机频率
                # basis_function_types.append(f'X_{i}_sine')
                # tanh项
                basis_functions.append(np.tanh(self.X[:, i]))
                # basis_function_types.append(f'X_{i}_tanh')

            # 3. 交叉项
            if self.n_features >= 2 and n_interaction_terms > 0:
                max_possible_interactions = self.n_features * (self.n_features - 1) // 2
                actual_interaction_terms = min(n_interaction_terms, max_possible_interactions)
                
                all_feature_pairs = [(i, j) for i in range(self.n_features) for j in range(i + 1, self.n_features)]
                chosen_pairs = set()

                if actual_interaction_terms < max_possible_interactions:
                    if len(all_feature_pairs) > 0: # Ensure all_feature_pairs is not empty
                        sampled_pairs_indices = self.rng.choice(len(all_feature_pairs), actual_interaction_terms, replace=False)
                        chosen_pairs = {all_feature_pairs[k] for k in sampled_pairs_indices}
                elif actual_interaction_terms > 0: # Use all possible combinations if actual_interaction_terms == max_possible_interactions
                    chosen_pairs = set(all_feature_pairs)

                for idx1, idx2 in chosen_pairs:
                    basis_functions.append(self.X[:, idx1] * self.X[:, idx2])
                    # basis_function_types.append(f'X_{idx1}*X_{idx2}_interaction')

            # 将所有基础函数堆叠成一个矩阵
            # 移除可能重复的或空的函数 (例如当 n_features 很小时)
            valid_basis_functions = [f for f in basis_functions if f.ndim == 1 and f.size == self.n_samples_total]
            if not valid_basis_functions:
                 # Fallback or error if no valid basis functions can be formed
                 # This might happen if n_samples_total is 0, or other extreme conditions
                 # For now, let's assume it results in a simple signal (e.g., zeros) if no basis.
                 # Or, more robustly, ensure there's always at least one basic feature if n_features > 0
                 if self.n_features > 0: # Add at least one raw feature if basis list is empty
                     basis_matrix = self.X[:, [0]] # Use first feature as a minimal basis
                 else: # If no features, then signal is zero
                     basis_matrix = np.zeros((self.n_samples_total, 1)) 
            else:
                basis_matrix = np.column_stack(valid_basis_functions)

            if basis_matrix.shape[1] == 0 and self.n_features > 0: # Should be caught above, but as a safeguard
                 basis_matrix = self.X[:, [0]] # Default to first feature
            elif basis_matrix.shape[1] == 0 and self.n_features == 0:
                 basis_matrix = np.zeros((self.n_samples_total, 1))


            # --- 为均值项和方差项生成系数 ---
            n_basis_funcs = basis_matrix.shape[1]
            if n_basis_funcs == 0: # Should not happen if safeguards above work
                n_basis_funcs = 1 # Avoid errors with empty basis_matrix
                basis_matrix = np.zeros((self.n_samples_total, 1)) # Default basis

            # 1. 均值项的系数
            mean_coeffs = self.rng.uniform(-1.5, 1.5, n_basis_funcs) * main_effect_strength
            
            # 2. 方差项（对数标准差）的系数
            log_std_coeffs = self.rng.uniform(-1.0, 1.0, n_basis_funcs) * hetero_strength

            # --- 计算均值项 (Y_signal) ---
            y_signal = np.dot(basis_matrix, mean_coeffs)
            
            # --- 计算异方差噪声的标准差 ---
            log_std_modulator = np.dot(basis_matrix, log_std_coeffs)
            
            if np.std(log_std_modulator) > 1e-6 : # Avoid division by zero if modulator is constant
                log_std_modulator = (log_std_modulator - np.mean(log_std_modulator)) / np.std(log_std_modulator)
            else: # If constant, normalize to 0
                log_std_modulator = np.zeros_like(log_std_modulator)

            log_std_modulator = np.clip(log_std_modulator, -3, 3) 
            
            std_multiplier = 0.1 + np.exp(log_std_modulator) 
            
            heteroscedastic_noise = noise_level * std_multiplier * self.rng.randn(self.n_samples_total)
            
            self.y = y_signal + heteroscedastic_noise
        
        else:
            raise ValueError(f"不支持的非线性函数类型: {function_type}")
        
        return self.X, self.y

    def inject_y_outliers(self, y_data, outlier_ratio=0.1, outlier_strength=5.0, y_outlier_method='additive_std'):
        """
        在目标变量y中注入异常值 (操作传入的 y_data 副本)
        """
        y_outliers_copy = y_data.copy().astype(float)
        n_samples_current = len(y_outliers_copy)
        n_outliers = int(n_samples_current * outlier_ratio)

        if n_outliers == 0:
            return y_outliers_copy, np.zeros(n_samples_current, dtype=bool)
            
        outlier_indices = self.rng.choice(n_samples_current, n_outliers, replace=False)
        outlier_mask = np.zeros(n_samples_current, dtype=bool)
        outlier_mask[outlier_indices] = True
        
        y_std_current = np.std(y_data) # 使用原始传入的 y_data 计算标准差

        if y_outlier_method in ['multiplicative', 'sequential_multiplicative_additive'] and outlier_strength == 0:
            raise ValueError(f"outlier_strength cannot be 0 for y_outlier_method '{y_outlier_method}'.")

        if y_outlier_method == 'additive_std':
            for idx in outlier_indices:
                if self.rng.rand() < 0.5:
                    y_outliers_copy[idx] += outlier_strength * y_std_current
                else:
                    y_outliers_copy[idx] -= outlier_strength * y_std_current
            
        elif y_outlier_method == 'multiplicative':
            for idx in outlier_indices:
                if self.rng.rand() < 0.5:
                    y_outliers_copy[idx] *= outlier_strength
                else:
                    if outlier_strength == 0: # Should be caught by earlier check, but good for safety
                         y_outliers_copy[idx] = 0 # Or handle as error
                    else:
                        y_outliers_copy[idx] /= outlier_strength
                        
        elif y_outlier_method == 'sequential_multiplicative_additive':
            for idx in outlier_indices:
                current_y_val = y_outliers_copy[idx]
                if self.rng.rand() < 0.5:
                    intermediate_y = current_y_val * outlier_strength
                else:
                    if outlier_strength == 0:
                        intermediate_y = 0
                    else:
                        intermediate_y = current_y_val / outlier_strength
                
                if self.rng.rand() < 0.5:
                    y_outliers_copy[idx] = intermediate_y + (outlier_strength * y_std_current)
                else:
                    y_outliers_copy[idx] = intermediate_y - (outlier_strength * y_std_current)
            
        else:
            raise ValueError(f"不支持的 y_outlier_method: {y_outlier_method}")
                
        return y_outliers_copy, outlier_mask

    def inject_x_outliers(self, X_data, outlier_ratio=0.1, outlier_strength=5.0):
        """
        在特征矩阵X中注入异常值（杠杆点） (操作传入的 X_data 副本)
        """
        X_outliers_copy = X_data.copy()
        n_samples_current, n_features_current = X_outliers_copy.shape
        n_outliers = int(n_samples_current * outlier_ratio)

        if n_outliers == 0:
            return X_outliers_copy, np.zeros(n_samples_current, dtype=bool)
        
        outlier_indices = self.rng.choice(n_samples_current, n_outliers, replace=False)
        outlier_mask = np.zeros(n_samples_current, dtype=bool)
        outlier_mask[outlier_indices] = True
        
        X_std_current = np.std(X_data, axis=0) # 使用原始传入的 X_data 计算标准差
        
        for idx in outlier_indices:
            feature_idx = self.rng.randint(0, n_features_current)
            # Ensure X_std_current[feature_idx] is not zero to prevent issues, though unlikely with randn
            std_val = X_std_current[feature_idx] if X_std_current[feature_idx] > 1e-9 else 1.0

            if self.rng.rand() > 0.5:
                X_outliers_copy[idx, feature_idx] += outlier_strength * std_val
            else:
                X_outliers_copy[idx, feature_idx] -= outlier_strength * std_val
        
        return X_outliers_copy, outlier_mask

# 模块级辅助函数
def split_data(X, y, test_size=0.15, val_size=0.15, random_state=None):
    """
    将数据集划分为训练集、验证集和测试集
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if (1 - test_size) <= 0: # Avoid division by zero if test_size is 1 or more
        if val_size > 0 :
             raise ValueError("Cannot create validation set if test_size is 1 or greater.")
        else: # No validation set needed
            return X_temp, None, X_test, y_temp, None, y_test

    val_ratio_in_temp = val_size / (1 - test_size)
    if val_ratio_in_temp >= 1.0 and val_size > 0: # val_size makes the temp set fully validation or more
        if val_size == (1-test_size): # temp set is entirely validation
             return np.array([]).reshape(0, X.shape[1]) if X.ndim > 1 else np.array([]), \
                    X_temp, X_test, \
                    np.array([]), y_temp, y_test
        else: # val_size is too large
            raise ValueError(f"val_size ({val_size}) is too large for the remaining data after test split ({1-test_size}).")
    
    if val_size == 0: # No validation set
        return X_temp, None, X_test, y_temp, None, y_test

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_in_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_synthetic_experiment(
    n_samples_total=1000, n_features=10, 
    outlier_ratio=0.1, outlier_strength=5.0, outlier_type='y', 
    relation_type='linear', 
    y_outlier_method='additive_std',
    # Parameters for generate_nonlinear if relation_type='interactive_heteroscedastic'
    n_interaction_terms=3, 
    interaction_strength=0.5,
    hetero_strength=0.5,
    main_effect_strength=1.0,
    noise_level_linear=1.0, # Specific noise for linear
    noise_level_nonlinear=0.5, # Specific noise for nonlinear
    random_state=None,
    # split_data params
    test_size=0.15, val_size=0.15
):
    """
    准备合成数据实验 (使用 SyntheticDataGenerator)
    """
    # 保持原始的随机状态设置方式，传递给生成器和split_data
    # generator 的 random_state 会初始化其内部的 rng
    # split_data 也会使用传入的 random_state
    
    generator = SyntheticDataGenerator(
        n_samples_total=n_samples_total, 
        n_features=n_features, 
        random_state=random_state
    )
    
    if relation_type == 'linear':
        X, y = generator.generate_linear(noise_level=noise_level_linear)
    elif relation_type in ['polynomial', 'sine', 'exp', 'interactive_heteroscedastic']:
        X, y = generator.generate_nonlinear(
            function_type=relation_type, 
            noise_level=noise_level_nonlinear,
            n_interaction_terms=n_interaction_terms,
            interaction_strength=interaction_strength,
            hetero_strength=hetero_strength,
            main_effect_strength=main_effect_strength
        )
    else:
        raise ValueError(f"不支持的关系类型: {relation_type}")
    
    # 划分数据 (使用模块级 split_data)
    X_train_orig, X_val_orig, X_test, y_train_orig, y_val_orig, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # 初始化变量，以防没有验证集或异常值
    y_train_processed = y_train_orig.copy() if y_train_orig is not None else None
    X_train_processed = X_train_orig.copy() if X_train_orig is not None else None
    y_val_processed = y_val_orig.copy() if y_val_orig is not None else None
    X_val_processed = X_val_orig.copy() if X_val_orig is not None else None
    
    outlier_mask_train = np.zeros(len(y_train_orig), dtype=bool) if y_train_orig is not None else None

    if X_val_orig is not None and y_val_orig is not None :
        outlier_mask_val = np.zeros(len(y_val_orig), dtype=bool)
    else:
        outlier_mask_val = None

    # 注入异常值（仅在训练集和验证集中，如果它们存在）
    # 使用生成器实例的 inject 方法，并传递对应的 rng 确保随机性一致（如果需要独立于生成器的，可以新创 RandomState）
    # 当前设计中，inject_* 方法使用 self.rng, 与生成器初始化时一致

    if outlier_ratio > 0:
        if outlier_type == 'y':
            if X_train_orig is not None and y_train_orig is not None:
                y_train_processed, outlier_mask_train = generator.inject_y_outliers(
                    y_train_orig, outlier_ratio, outlier_strength, y_outlier_method
                )
            if X_val_orig is not None and y_val_orig is not None:
                y_val_processed, outlier_mask_val = generator.inject_y_outliers(
                    y_val_orig, outlier_ratio, outlier_strength, y_outlier_method
                )
        
        elif outlier_type == 'x':
            if X_train_orig is not None and y_train_orig is not None:
                 X_train_processed, outlier_mask_train = generator.inject_x_outliers(
                    X_train_orig, outlier_ratio, outlier_strength
                )
            if X_val_orig is not None and y_val_orig is not None:
                X_val_processed, outlier_mask_val = generator.inject_x_outliers(
                    X_val_orig, outlier_ratio, outlier_strength
                )
    
    return X_train_processed, X_val_processed, X_test, \
           y_train_processed, y_val_processed, y_test, \
           outlier_mask_train, outlier_mask_val

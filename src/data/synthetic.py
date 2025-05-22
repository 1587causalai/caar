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

def generate_linear_data(n_samples=1000, n_features=10, noise_level=0.5, random_state=None):
    """
    生成具有线性关系的合成数据
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        noise_level: 噪声水平
        random_state: 随机种子
        
    返回:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,)
        true_coef: 真实系数，形状为 (n_features,)
    """
    # 设置随机种子
    rng = np.random.RandomState(random_state)
    
    # 生成特征矩阵
    X = rng.randn(n_samples, n_features)
    
    # 生成真实系数
    true_coef = rng.randn(n_features)
    
    # 生成目标变量: y = X * true_coef + noise
    y = np.dot(X, true_coef) + noise_level * rng.randn(n_samples)
    
    return X, y, true_coef

def generate_nonlinear_data(n_samples=1000, n_features=10, noise_level=0.5, 
                            function_type='polynomial', 
                            n_interaction_terms=3, 
                            interaction_strength=0.5,
                            hetero_strength=0.5,
                            main_effect_strength=1.0,
                            random_state=None):
    """
    生成具有非线性关系的合成数据
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        noise_level: 基础噪声水平
        function_type: 非线性函数类型，可选值为 'polynomial', 'sine', 'exp', 'interactive_heteroscedastic'
        n_interaction_terms: (仅当 function_type='interactive_heteroscedastic' 时) 生成的特征交叉项的数量。
        interaction_strength: (仅当 function_type='interactive_heteroscedastic' 时) 交叉项的系数大小。
        hetero_strength: (仅当 function_type='interactive_heteroscedastic' 时) 控制异方差噪声强度的参数。
        main_effect_strength: (仅当 function_type='interactive_heteroscedastic' 时) 控制主要非线性效应的系数大小。
        random_state: 随机种子
        
    返回:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,)
    """
    # 设置随机种子
    rng = np.random.RandomState(random_state)
    
    # 生成特征矩阵
    X = rng.randn(n_samples, n_features)
    
    # 生成目标变量
    if function_type == 'polynomial':
        # 多项式关系: y = sum(x_i^2) + noise
        y = np.sum(X**2, axis=1) + noise_level * rng.randn(n_samples)
    
    elif function_type == 'sine':
        # 正弦关系: y = sin(sum(x_i)) + noise
        y = np.sin(np.sum(X, axis=1)) + noise_level * rng.randn(n_samples)
    
    elif function_type == 'exp':
        # 指数关系: y = exp(sum(x_i)) + noise
        # 为避免数值溢出，对X进行缩放
        y = np.exp(np.sum(X * 0.1, axis=1)) + noise_level * rng.randn(n_samples)

    elif function_type == 'interactive_heteroscedastic':
        if n_features < 2 and n_interaction_terms > 0:
            raise ValueError("Cannot create interaction terms with less than 2 features.")
        
        # 1. 基础非线性信号 (main effects)
        # 使用一些非线性变换，例如 sin 和多项式项的组合
        y_base = np.zeros(n_samples)
        # 为每个特征随机分配权重
        main_coeffs = rng.uniform(-1.5, 1.5, n_features) * main_effect_strength
        for i in range(n_features):
            # 可以混合不同的非线性变换
            if i % 3 == 0:
                y_base += main_coeffs[i] * np.sin(X[:, i] * rng.uniform(0.5, 1.5))
            elif i % 3 == 1:
                y_base += main_coeffs[i] * (X[:, i]**2) * rng.uniform(0.2, 0.8)
            else:
                y_base += main_coeffs[i] * np.tanh(X[:, i])

        # 2. 特征交叉项
        y_interactions = np.zeros(n_samples)
        if n_features >= 2 and n_interaction_terms > 0:
            actual_interaction_terms = min(n_interaction_terms, n_features * (n_features - 1) // 2) # 确保不超过可能的组合数
            interaction_coeffs = rng.uniform(-1, 1, actual_interaction_terms) * interaction_strength
            
            # 随机选择特征对进行交叉
            chosen_pairs = set()
            for k in range(actual_interaction_terms):
                # 确保选取的特征对不重复
                while True:
                    idx1, idx2 = rng.choice(n_features, 2, replace=False)
                    pair = tuple(sorted((idx1, idx2)))
                    if pair not in chosen_pairs:
                        chosen_pairs.add(pair)
                        break
                y_interactions += interaction_coeffs[k] * X[:, idx1] * X[:, idx2]

        # 3. 组合信号
        y_signal = y_base + y_interactions
        
        # 4. 异方差噪声
        # 噪声的方差由一个或多个特征的函数决定
        # 例如，让标准差与第一个特征的绝对值的指数函数相关，并添加基础噪声水平
        noise_variance_modulator = np.exp(hetero_strength * np.abs(X[:, 0])) 
        # 可以将调制因子进行标准化或限制，以避免极端值
        noise_variance_modulator = (noise_variance_modulator - np.min(noise_variance_modulator)) / \
                                   (np.max(noise_variance_modulator) - np.min(noise_variance_modulator) + 1e-6) # 归一化到 [0,1]
        noise_variance_modulator = 0.5 + 1.5 * noise_variance_modulator # 调整范围，例如到 [0.5, 2.0]

        heteroscedastic_noise = noise_level * noise_variance_modulator * rng.randn(n_samples)
        
        y = y_signal + heteroscedastic_noise
    
    else:
        raise ValueError(f"不支持的非线性函数类型: {function_type}")
    
    return X, y

def inject_y_outliers(X, y, outlier_ratio=0.1, outlier_strength=5.0, y_outlier_method='additive_std', random_state=None):
    """
    在目标变量y中注入异常值
    
    参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,)
        outlier_ratio: 异常值比例
        outlier_strength: 异常值强度.
                          对于 'additive_std', 这是标准差的倍数.
                          对于 'multiplicative', 这是乘法/除法因子.
                          对于 'sequential_multiplicative_additive', 这是乘法/除法因子和标准差的倍数.
        y_outlier_method: 注入y异常值的方法。可选值为:
                          'additive_std': y_new = y_original +/- outlier_strength * std(y_original) (随机选择加或减)
                          'multiplicative': y_new = y_original * outlier_strength OR y_original / outlier_strength (随机选择乘或除)
                          'sequential_multiplicative_additive': y_new = (y_original * or / outlier_strength) +/- (outlier_strength * std(y_original)) (每步都随机选择)
        random_state: 随机种子
        
    返回:
        X: 原始特征矩阵
        y_outliers: 包含异常值的目标变量
        outlier_mask: 异常值掩码，形状为 (n_samples,)，True表示异常值
    """
    # 设置随机种子
    rng = np.random.RandomState(random_state)
    
    # 复制原始目标变量
    y_outliers = y.copy().astype(float) # 确保是float类型以支持各种操作
    
    # 样本数量
    n_samples = len(y)
    
    # 异常值数量
    n_outliers = int(n_samples * outlier_ratio)
    if n_outliers == 0:
        return X, y_outliers, np.zeros(n_samples, dtype=bool)
        
    # 随机选择异常值索引
    outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
    
    # 创建异常值掩码
    outlier_mask = np.zeros(n_samples, dtype=bool)
    outlier_mask[outlier_indices] = True
    
    y_std = np.std(y_outliers) # 计算原始y的标准差 (在未修改的y_outliers上计算)

    if y_outlier_method in ['multiplicative', 'sequential_multiplicative_additive'] and outlier_strength == 0:
        raise ValueError(f"outlier_strength cannot be 0 for y_outlier_method '{y_outlier_method}'.")

    if y_outlier_method == 'additive_std':
        for idx in outlier_indices:
            if rng.rand() < 0.5:
                y_outliers[idx] += outlier_strength * y_std
            else:
                y_outliers[idx] -= outlier_strength * y_std
        
    elif y_outlier_method == 'multiplicative':
        for idx in outlier_indices:
            if rng.rand() < 0.5: # 50% chance to multiply
                y_outliers[idx] *= outlier_strength
            else: # 50% chance to divide
                y_outliers[idx] /= outlier_strength # outlier_strength is confirmed not 0
                    
    elif y_outlier_method == 'sequential_multiplicative_additive':
        for idx in outlier_indices:
            current_y_val = y_outliers[idx]
            
            # 步骤1: 乘/除部分 (随机选择)
            if rng.rand() < 0.5:
                intermediate_y = current_y_val * outlier_strength
            else:
                intermediate_y = current_y_val / outlier_strength # outlier_strength 确认非0
            
            # 步骤2: 加/减部分 (随机选择)
            if rng.rand() < 0.5:
                y_outliers[idx] = intermediate_y + (outlier_strength * y_std)
            else:
                y_outliers[idx] = intermediate_y - (outlier_strength * y_std)
        
    else:
        raise ValueError(f"不支持的 y_outlier_method: {y_outlier_method}")
            
    return X, y_outliers, outlier_mask

def inject_x_outliers(X, y, outlier_ratio=0.1, outlier_strength=5.0, random_state=None):
    """
    在特征矩阵X中注入异常值（杠杆点）
    
    参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 目标变量，形状为 (n_samples,)
        outlier_ratio: 异常值比例
        outlier_strength: 异常值强度，表示异常值偏离正常值的标准差倍数
        random_state: 随机种子
        
    返回:
        X_outliers: 包含异常值的特征矩阵
        y: 原始目标变量
        outlier_mask: 异常值掩码，形状为 (n_samples,)，True表示异常值
    """
    # 设置随机种子
    rng = np.random.RandomState(random_state)
    
    # 复制原始特征矩阵
    X_outliers = X.copy()
    
    # 样本数量和特征数量
    n_samples, n_features = X.shape
    
    # 异常值数量
    n_outliers = int(n_samples * outlier_ratio)
    
    # 随机选择异常值索引
    outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
    
    # 创建异常值掩码
    outlier_mask = np.zeros(n_samples, dtype=bool)
    outlier_mask[outlier_indices] = True
    
    # 计算每个特征的标准差
    X_std = np.std(X, axis=0)
    
    # 对于每个选定的样本，随机选择一个特征进行异常值注入
    for idx in outlier_indices:
        # 随机选择一个特征
        feature_idx = rng.randint(0, n_features)
        
        # 注入异常值
        # 随机决定是向上还是向下偏移
        if rng.rand() > 0.5:
            X_outliers[idx, feature_idx] += outlier_strength * X_std[feature_idx]
        else:
            X_outliers[idx, feature_idx] -= outlier_strength * X_std[feature_idx]
    
    return X_outliers, y, outlier_mask

def split_data(X, y, test_size=0.15, val_size=0.15, random_state=None):
    """
    将数据集划分为训练集、验证集和测试集
    
    参数:
        X: 特征矩阵
        y: 目标变量
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        
    返回:
        X_train: 训练集特征
        X_val: 验证集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_val: 验证集标签
        y_test: 测试集标签
    """
    # 首先划分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 从剩余数据中划分出验证集
    # 计算验证集在剩余数据中的比例
    val_ratio = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_synthetic_experiment(n_samples=1000, n_features=10, outlier_ratio=0.1, 
                                outlier_strength=5.0, outlier_type='y', 
                                relation_type='linear', 
                                y_outlier_method='additive_std',
                                random_state=None):
    """
    准备合成数据实验
    
    参数:
        n_samples: 样本数量
        n_features: 特征数量
        outlier_ratio: 异常值比例
        outlier_strength: 异常值强度
        outlier_type: 异常值类型，可选值为 'y', 'x', 'none'
        relation_type: 关系类型，可选值为 'linear', 'polynomial', 'sine', 'exp', 'interactive_heteroscedastic'
        y_outlier_method: 当outlier_type='y'时，注入y异常值的方法。
                          参见 inject_y_outliers 的文档.
        random_state: 随机种子
        
    返回:
        X_train: 训练集特征
        X_val: 验证集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_val: 验证集标签
        y_test: 测试集标签
        outlier_mask_train: 训练集异常值掩码
        outlier_mask_val: 验证集异常值掩码
    """
    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)
    
    # 生成数据
    if relation_type == 'linear':
        X, y, _ = generate_linear_data(n_samples, n_features, random_state=random_state)
    elif relation_type in ['polynomial', 'sine', 'exp', 'interactive_heteroscedastic']:
        X, y = generate_nonlinear_data(n_samples, n_features, function_type=relation_type, random_state=random_state)
    else:
        raise ValueError(f"不支持的关系类型: {relation_type}")
    
    # 划分数据
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, random_state=random_state
    )
    
    # 初始化异常值掩码
    outlier_mask_train = np.zeros(len(y_train), dtype=bool)
    outlier_mask_val = np.zeros(len(y_val), dtype=bool)
    
    # 注入异常值（仅在训练集和验证集中）
    if outlier_type == 'y' and outlier_ratio > 0:
        # 在训练集中注入Y异常值
        _, y_train, outlier_mask_train = inject_y_outliers(
            X_train, y_train, outlier_ratio, outlier_strength, 
            y_outlier_method=y_outlier_method, random_state=random_state
        )
        
        # 在验证集中注入Y异常值
        _, y_val, outlier_mask_val = inject_y_outliers(
            X_val, y_val, outlier_ratio, outlier_strength, 
            y_outlier_method=y_outlier_method, random_state=random_state
        )
    
    elif outlier_type == 'x' and outlier_ratio > 0:
        # 在训练集中注入X异常值
        X_train, _, outlier_mask_train = inject_x_outliers(
            X_train, y_train, outlier_ratio, outlier_strength, random_state=random_state
        )
        
        # 在验证集中注入X异常值
        X_val, _, outlier_mask_val = inject_x_outliers(
            X_val, y_val, outlier_ratio, outlier_strength, random_state=random_state
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, outlier_mask_train, outlier_mask_val

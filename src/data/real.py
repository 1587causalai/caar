"""
真实数据加载模块

提供以下功能：
1. 加载常见真实回归数据集
2. 在真实数据集中注入异常值
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_california_housing():
    """
    加载California Housing数据集
    
    返回:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称
    """
    # 加载数据集
    housing = fetch_california_housing(as_frame=True)
    
    # 获取特征矩阵和目标变量
    X = housing.data.values
    y = housing.target.values
    
    # 获取特征名称
    feature_names = housing.feature_names
    
    return X, y, feature_names

def load_diabetes_dataset():
    """
    加载Diabetes数据集
    
    返回:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称
    """
    # 加载数据集
    diabetes = load_diabetes(as_frame=True)
    
    # 获取特征矩阵和目标变量
    X = diabetes.data.values
    y = diabetes.target.values
    
    # 获取特征名称
    feature_names = diabetes.feature_names
    
    return X, y, feature_names

def preprocess_data(X_train, X_val, X_test, scale=True):
    """
    预处理数据（标准化）
    
    参数:
        X_train: 训练集特征
        X_val: 验证集特征
        X_test: 测试集特征
        scale: 是否进行标准化
        
    返回:
        X_train_scaled, X_val_scaled, X_test_scaled: 标准化后的特征矩阵
        scaler: 标准化器（如果scale=True）
    """
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    return X_train, X_val, X_test, scaler

def inject_outliers_to_real_data(X, y, outlier_ratio=0.1, outlier_strength=5.0, outlier_type='y', y_outlier_method='additive_std', random_state=None):
    """
    在真实数据集中注入异常值
    
    参数:
        X: 特征矩阵
        y: 目标变量
        outlier_ratio: 异常值比例
        outlier_strength: 异常值强度.
                          对于 'additive_std', 这是标准差的倍数.
                          对于 'multiplicative', 这是乘法/除法因子.
                          对于 'sequential_multiplicative_additive', 这是乘法/除法因子和标准差的倍数.
        outlier_type: 异常值类型，可选值为 'y', 'x'
        y_outlier_method: 当outlier_type='y'时，注入y异常值的方法。可选值为:
                          'additive_std': y_new = y_original +/- outlier_strength * std(y_original)
                          'multiplicative': y_new = y_original * outlier_strength OR y_original / outlier_strength
                          'sequential_multiplicative_additive': y_new = (y_original * or / outlier_strength) +/- (outlier_strength * std(y_original))
        random_state: 随机种子
        
    返回:
        X_outliers: 包含异常值的特征矩阵（如果outlier_type='x'或未更改）
        y_outliers: 包含异常值的目标变量（如果outlier_type='y'）
        outlier_mask: 异常值掩码
    """
    # 设置随机种子
    rng = np.random.RandomState(random_state)
    
    # 样本数量
    n_samples = len(y)
    
    # 异常值数量
    n_outliers = int(n_samples * outlier_ratio)
    
    # 随机选择异常值索引
    outlier_indices = rng.choice(n_samples, n_outliers, replace=False)
    
    # 创建异常值掩码
    outlier_mask = np.zeros(n_samples, dtype=bool)
    outlier_mask[outlier_indices] = True
    
    if outlier_type == 'y':
        # 复制原始目标变量
        y_outliers = y.copy().astype(float)
        y_std = np.std(y_outliers) # 计算原始y的标准差

        if y_outlier_method in ['multiplicative', 'sequential_multiplicative_additive'] and outlier_strength == 0:
            raise ValueError(f"outlier_strength cannot be 0 for y_outlier_method '{y_outlier_method}' when outlier_type is 'y'.")

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
    
    elif outlier_type == 'x':
        # 复制原始特征矩阵
        X_outliers = X.copy()
        
        # 特征数量
        n_features = X.shape[1]
        
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
    
    else:
        raise ValueError(f"不支持的异常值类型: {outlier_type}")

def prepare_real_data_experiment(dataset_name, data_dir='./data', outlier_ratio=0.1, outlier_strength=5.0, 
                               outlier_type='y', y_outlier_method='additive_std', test_size=0.15, val_size=0.15, 
                               scale=True, random_state=None):
    """
    准备真实数据实验
    
    参数:
        dataset_name: 数据集名称，可选值为 'california', 'diabetes', 'boston_housing', 
                                        'concrete_strength', 'communities_crime', 'bike_sharing', 'parkinsons_telemonitoring'
        data_dir: 存放本地数据文件的目录
        outlier_ratio: 异常值比例
        outlier_strength: 异常值强度
        outlier_type: 异常值类型，可选值为 'y', 'x', 'none'
        y_outlier_method: 当outlier_type='y'时，注入y异常值的方法。
                          参见 inject_outliers_to_real_data 的文档.
        test_size: 测试集比例
        val_size: 验证集比例
        scale: 是否进行标准化
        random_state: 随机种子
        
    返回:
        X_train: 训练集特征
        X_val: 验证集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_val: 验证集标签
        y_test: 测试集标签
        feature_names: 特征名称
        outlier_mask_train: 训练集异常值掩码
        outlier_mask_val: 验证集异常值掩码
        scaler: 标准化器（如果scale=True）
    """
    # 加载数据集
    if dataset_name == 'california':
        X, y, feature_names = load_california_housing()
    elif dataset_name == 'diabetes':
        X, y, feature_names = load_diabetes_dataset()
    elif dataset_name == 'boston_housing':
        boston = fetch_openml(data_id=531, as_frame=True, parser='auto')
        X_df = boston.frame[boston.feature_names]
        y_series = boston.frame[boston.target_names[0]]
        X = X_df.values
        y = y_series.values
        feature_names = list(X_df.columns)
    elif dataset_name == 'concrete_strength':
        xls_path = os.path.join(data_dir, 'Concrete_Data.xls')
        csv_path = os.path.join(data_dir, 'Concrete_Data.csv')
        if os.path.exists(xls_path):
            file_path = xls_path
            df = pd.read_excel(file_path)
        elif os.path.exists(csv_path):
            file_path = csv_path
            df = pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"Neither Concrete_Data.xls nor Concrete_Data.csv found in '{data_dir}'. Please download from UCI.")
        X_df = df.iloc[:, :-1]
        y_series = df.iloc[:, -1]
        X = X_df.values
        y = y_series.values
        feature_names = list(X_df.columns)
    elif dataset_name == 'communities_crime':
        file_path = os.path.join(data_dir, 'communities.data')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found. Please download from UCI and place it in the '{data_dir}' directory.")
        # Column names are not in the file, refer to communities.names
        # For simplicity, we'll assume a common setup: skip first 5 non-predictive, last is target
        # Actual column names/indices might vary based on the specific version of the .names file.
        # This dataset often has many missing values marked as '?'
        df = pd.read_csv(file_path, header=None, na_values='?')
        
        # Drop non-predictive leading columns (e.g., state, county, community, communityname, fold) - typically first 5
        # The actual number of columns to skip and the target column index should be verified from communities.names
        # For this simplified version, we assume target is the last column, and we skip 5 leading columns.
        # A more robust parser would use information from communities.names to select features.
        
        # Crude way to get potential feature columns and target
        # Assuming columns 0-4 are non-predictive, and the last column is the target
        # This is a strong assumption and might need adjustment
        if df.shape[1] > 5: # Ensure there are enough columns
            # Identify target column (e.g., 'ViolentCrimesPerPop' which is often the last or near last)
            # For this example, let's assume the last column is the target.
            y_series = df.iloc[:, -1].astype(float) # Attempt to convert target to float
            
            # Features are columns between the non-predictive ones and the target
            # This is a rough selection, many columns might be non-numeric or irrelevant
            X_df_potential = df.iloc[:, 5:-1] 
            
            # Attempt to convert all feature columns to numeric, errors will become NaN
            X_df_numeric = X_df_potential.apply(pd.to_numeric, errors='coerce')
            
            # Handle missing values in y: drop rows where y is NaN
            valid_y_indices = ~y_series.isna()
            y = y_series[valid_y_indices].values
            X_df_cleaned_rows = X_df_numeric[valid_y_indices]

            # Handle missing values in X: fill with mean for each column
            X_filled = X_df_cleaned_rows.fillna(X_df_cleaned_rows.mean())
            
            # If any column is still all NaN (e.g., all values were non-numeric and no mean could be calculated), fill with 0
            X_filled = X_filled.fillna(0)

            X = X_filled.values
            feature_names = [f'feature_{i}' for i in range(X.shape[1])] # Generic feature names
        else:
            raise ValueError(f"Communities and Crime data file '{file_path}' does not have enough columns for the assumed structure.")
    elif dataset_name == 'bike_sharing':
        file_path = os.path.join(data_dir, 'hour.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found. Please run download_data.py or place it in the '{data_dir}' directory.")
        df = pd.read_csv(file_path)
        # Drop unnecessary columns
        # 'instant' is just an index
        # 'dteday' is a date, which is already represented by yr, mnth, hr etc.
        # 'casual' and 'registered' sum up to 'cnt'. We use 'cnt' as target, so drop these to avoid leakage and redundancy.
        df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])
        
        y_series = df['cnt']
        X_df = df.drop(columns=['cnt'])
        
        X = X_df.values
        y = y_series.values
        feature_names = list(X_df.columns)
    elif dataset_name == 'parkinsons_telemonitoring':
        file_path = os.path.join(data_dir, 'parkinsons_updrs.data')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found. Please run download_data.py or place it in the '{data_dir}' directory.")
        df = pd.read_csv(file_path)
        # Target variable: total_UPDRS (column index 5)
        # Features: age (col 1), sex (col 2), Jitter ... PPE (cols 6-21)
        # Drop: subject# (col 0), test_time (col 3), motor_UPDRS (col 4)
        
        # Correct column indexing based on typical CSV reading (0-indexed)
        # subject#,age,sex,test_time,motor_UPDRS,total_UPDRS,Jitter(%)
        #   0       1   2     3           4           5        6...

        y_series = df.iloc[:, 5] # total_UPDRS
        
        # Select feature columns: age, sex, and then Jitter onwards
        # Columns for features: 1 (age), 2 (sex), 6 to end (Jitter, Shimmer etc.)
        X_cols_part1 = df.iloc[:, [1, 2]] # age, sex
        X_cols_part2 = df.iloc[:, 6:]     # Jitter onwards
        X_df = pd.concat([X_cols_part1, X_cols_part2], axis=1)
        
        X = X_df.values
        y = y_series.values
        feature_names = list(X_df.columns)
            
    else:
        raise ValueError(f"不支持的数据集名称: {dataset_name}")
    
    # 划分数据
    # 首先划分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 从剩余数据中划分出验证集
    # 计算验证集在剩余数据中的比例
    val_ratio = val_size / (1 - test_size)
    
    X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    # 标准化 -  Moved here: fit on train, transform val and test
    X_train, X_val, X_test, scaler = preprocess_data(X_train_orig, X_val_orig, X_test, scale=scale)
    
    # y_train and y_val should correspond to the scaled X, so we use y_train_orig and y_val_orig
    y_train = y_train_orig
    y_val = y_val_orig

    # 初始化异常值掩码
    outlier_mask_train = np.zeros(len(y_train), dtype=bool)
    outlier_mask_val = np.zeros(len(y_val), dtype=bool)
    
    # 注入异常值（仅在训练集和验证集中）
    if outlier_type != 'none' and outlier_ratio > 0:
        # 在训练集中注入异常值
        X_train, y_train, outlier_mask_train = inject_outliers_to_real_data(
            X_train, y_train, outlier_ratio, outlier_strength, outlier_type, 
            y_outlier_method=y_outlier_method, random_state=random_state
        )
        
        # 在验证集中注入异常值
        # 为了确保验证集和训练集的异常值注入方式一致，但随机性可能不同（如果需要），这里也传递 y_outlier_method
        # 如果希望验证集的随机种子与训练集不同，可以为验证集传递不同的 random_state 或 None
        # 当前实现下，训练集和验证集使用相同的 random_state (如果提供) 来选择异常点索引，这通常是期望行为以保持可复现性
        X_val, y_val, outlier_mask_val = inject_outliers_to_real_data(
            X_val, y_val, outlier_ratio, outlier_strength, outlier_type, 
            y_outlier_method=y_outlier_method, random_state=random_state # 保持random_state一致，或按需调整
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, outlier_mask_train, outlier_mask_val, scaler

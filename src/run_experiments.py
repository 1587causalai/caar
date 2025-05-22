"""
主实验执行脚本

执行所有实验并生成结果报告
"""

import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
sys.path.append('/home/ubuntu/robust-regression-experiment/src')

# 导入实验模块
from experiments.synthetic_exp import run_all_synthetic_experiments
from experiments.real_exp import run_all_real_data_experiments

def create_results_directory():
    """创建结果目录"""
    results_dir = '/home/ubuntu/robust-regression-experiment/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建子目录
    os.makedirs(os.path.join(results_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    
    return results_dir

def run_experiments(results_dir):
    """运行所有实验"""
    print("开始执行所有实验...")
    start_time = time.time()
    
    # 记录实验开始时间
    with open(os.path.join(results_dir, 'experiment_log.txt'), 'w') as f:
        f.write(f"实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 运行合成数据实验
    print("\n=== 运行合成数据实验 ===")
    synthetic_results = run_all_synthetic_experiments(
        outlier_ratios=[0.0, 0.05, 0.1, 0.2],  # 简化比例列表以加快实验速度
        n_samples=500,  # 减少样本数量以加快实验速度
        n_features=5,   # 减少特征数量以加快实验速度
        n_repeats=3,    # 减少重复次数以加快实验速度
        results_dir=results_dir
    )
    
    # 运行真实数据实验
    print("\n=== 运行真实数据实验 ===")
    real_results = run_all_real_data_experiments(
        outlier_ratios=[0.0, 0.05, 0.1, 0.2],
        n_repeats=3,
        results_dir=results_dir
    )
    
    # 记录实验结束时间和总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    with open(os.path.join(results_dir, 'experiment_log.txt'), 'a') as f:
        f.write(f"实验结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)\n")
    
    print(f"\n所有实验完成！总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    
    # 返回所有结果
    return {
        'synthetic': synthetic_results,
        'real': real_results,
        'total_time': total_time
    }

def main():
    """主函数"""
    # 创建结果目录
    results_dir = create_results_directory()
    
    # 运行所有实验
    all_results = run_experiments(results_dir)
    
    # 保存所有结果
    with open(os.path.join(results_dir, 'all_experiments_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"所有实验结果已保存至: {os.path.join(results_dir, 'all_experiments_results.pkl')}")

if __name__ == "__main__":
    main()

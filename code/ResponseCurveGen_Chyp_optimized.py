# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 19:44:21 2025

@author: Leah
"""

import elapid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
from tqdm import tqdm

# 设置matplotlib后端
matplotlib.use('Agg')

# 忽略警告
warnings.filterwarnings("ignore")

def load_and_prepare_data(file_path):
    """加载和准备数据"""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # 数据验证
    required_columns = ['Chyp_c1c2', 'year', 'MLDaverSST', 'CHL', 'MLDaverCHL', 'SST', 'MLDaverSal', 'nitrat']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 创建目标变量
    df["Chyp_occur"] = (df["Chyp_c1c2"] != 0).astype(int)
    df["cycle_year"] = df['year']
    
    # 移除包含NaN的行
    feature_names = ['MLDaverSST', 'CHL', 'MLDaverCHL', 'SST', 'MLDaverSal', 'nitrat']
    df_clean = df.dropna(subset=feature_names + ['Chyp_occur'])
    
    print(f"Original data shape: {df.shape}")
    print(f"Clean data shape: {df_clean.shape}")
    
    return df_clean, feature_names

def calculate_response_curves(df, feature_names, cycle_years):
    """计算所有特征的响应曲线"""
    print("Calculating response curves...")
    
    # 初始化存储所有周期响应曲线的字典
    all_response_curves = {feature: [] for feature in feature_names}
    
    # 循环每个周期年份
    for cycle_year in tqdm(cycle_years, desc="Processing years"):
        try:
            # 分割训练集和测试集
            test_mask = df['cycle_year'] == cycle_year
            train_mask = ~test_mask
            
            if test_mask.sum() < 1:
                print(f"Warning: No test data for year {cycle_year}")
                continue
                
            X_train = df.loc[train_mask, feature_names]
            y_train = df.loc[train_mask, 'Chyp_occur']
            
            # 训练模型
            model = elapid.MaxentModel()
            model.fit(X_train, y_train)
            
            # 计算每个特征的响应曲线
            for feature in feature_names:
                # 创建评估范围
                min_val = df[feature].min()
                max_val = df[feature].max()
                values = np.linspace(min_val, max_val, 100)
                
                # 创建评估数据框 - 优化版本
                eval_data = X_train.median().to_frame().T
                eval_data = pd.concat([eval_data] * 100, ignore_index=True)
                eval_data[feature] = values
                
                # 预测
                predictions = model.predict(eval_data)
                
                # 存储响应曲线
                all_response_curves[feature].append(predictions)
                
        except Exception as e:
            print(f"Error processing year {cycle_year}: {e}")
            continue
    
    return all_response_curves

def plot_response_curves(all_response_curves, feature_names, df, output_path):
    """绘制响应曲线"""
    print("Plotting response curves...")
    
    # 计算平均响应曲线
    mean_response = {}
    x_values = {}
    
    for feature in feature_names:
        if all_response_curves[feature]:  # 确保有数据
            min_val = df[feature].min()
            max_val = df[feature].max()
            x_values[feature] = np.linspace(min_val, max_val, 100)
            mean_response[feature] = np.mean(all_response_curves[feature], axis=0)
        else:
            print(f"Warning: No response curves for feature {feature}")
    
    # 创建2行3列的子图
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    
    # 绘制每个特征的响应曲线
    for i, feature in enumerate(feature_names):
        ax = axes[i]
        
        if feature in mean_response and feature in x_values:
            # 绘制平均响应曲线
            ax.plot(x_values[feature], mean_response[feature], 'b-', linewidth=2)
            
            # 绘制所有周期响应曲线的范围
            if len(all_response_curves[feature]) > 1:
                lower = np.percentile(all_response_curves[feature], 5, axis=0)
                upper = np.percentile(all_response_curves[feature], 95, axis=0)
                ax.fill_between(x_values[feature], lower, upper, color='b', alpha=0.2)
            
            ax.set_title(feature)
            ax.set_xlabel(feature)
            ax.set_ylabel('Predicted probability')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, f'No data for {feature}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Response curves saved to: {output_path}")

def main():
    """主函数"""
    # 配置参数
    data_path = "Chyp_c1c2_mldaver.csv"
    output_path = "E:/ECNU/20250721/responseCurve_Chyp.jpg"
    
    try:
        # 加载数据
        df, feature_names = load_and_prepare_data(data_path)
        
        # 获取年份列表
        cycle_years = sorted(df['cycle_year'].unique())
        print(f"Available years: {cycle_years}")
        
        # 计算响应曲线
        all_response_curves = calculate_response_curves(df, feature_names, cycle_years)
        
        # 绘制响应曲线
        plot_response_curves(all_response_curves, feature_names, df, output_path)
        
        print("Response curve generation completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == '__main__':
    main() 
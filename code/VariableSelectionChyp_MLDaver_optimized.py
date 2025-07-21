# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:45:01 2025

@author: Yiyang Tan
"""

import elapid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import itertools
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, roc_auc_score, mean_squared_error, 
    confusion_matrix
)

# 设置matplotlib后端
matplotlib.use('Agg')

# 忽略警告
warnings.filterwarnings("ignore")

def calculate_tss(y_true, y_pred):
    """计算True Skill Statistic (TSS)"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    tss = sensitivity + specificity - 1
    return tss

def load_and_prepare_data(file_path):
    """加载和准备数据"""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # 数据验证
    required_columns = ['Chyp_c1c2', 'year', 'SIC', 'CHL', 'SST', 'Sal', 'nitrat', 'mlotst_glor', 
                       'MLDaverCHL', 'MLDaverSal', 'MLDaverSST']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 创建目标变量
    df["Chyp_occur"] = (df["Chyp_c1c2"] != 0).astype(int)
    df["cycle_year"] = df['year']
    
    # 移除包含NaN的行
    feature_names = ['SIC', 'CHL', 'SST', 'Sal', 'nitrat', 'mlotst_glor', 
                    'MLDaverCHL', 'MLDaverSal', 'MLDaverSST']
    df_clean = df.dropna(subset=feature_names + ['Chyp_occur'])
    
    print(f"Original data shape: {df.shape}")
    print(f"Clean data shape: {df_clean.shape}")
    
    return df_clean, feature_names

def calbestComb(num, df, feature_names, cycle_years):
    """计算最佳特征组合"""
    results = []
    total_combinations = len(list(itertools.combinations(feature_names, num)))
    
    for combination in tqdm(itertools.combinations(feature_names, num), 
                           total=total_combinations, desc=f"Testing {num}-feature combinations"):
        remaining_features = list(combination)
        mse_list = []

        for cycle_year in cycle_years:
            # 分割训练和测试数据
            test_mask = df['cycle_year'] == cycle_year
            train_mask = ~test_mask
            
            if test_mask.sum() < 1:
                continue
                
            X_train = df.loc[train_mask, remaining_features]
            y_train = df.loc[train_mask, 'Chyp_occur']
            X_test = df.loc[test_mask, remaining_features]
            y_test = df.loc[test_mask, 'Chyp_occur']
            
            # 训练模型
            model = elapid.MaxentModel()
            model.fit(X_train, y_train)
            
            # 预测和评估
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_list.append(mse)

        if mse_list:  # 确保有有效的MSE值
            mean_mse = np.mean(mse_list)
            results.append({
                'combination': combination,
                'mse': mean_mse
            })
        
    if not results:
        raise ValueError(f"No valid results for {num} features")
        
    results_df = pd.DataFrame(results)
    min_mse_row = results_df.loc[results_df['mse'].idxmin()]    
    return results_df, min_mse_row

def plotGCV(feature_names, sector, df, cycle_years):
    """绘制GCV结果"""
    num = len(feature_names)
    mse_list = []
    auc_list = []
    tss_list = []
    feature_contributions = []
    
    # ROC曲线图
    plt.figure(figsize=(6, 5))
    
    for cycle_year in cycle_years:
        # 分割数据
        test_mask = df['cycle_year'] == cycle_year
        train_mask = ~test_mask
        
        if test_mask.sum() < 1:
            continue
            
        X_train = df.loc[train_mask, feature_names]
        y_train = df.loc[train_mask, 'Chyp_occur']
        X_test = df.loc[test_mask, feature_names]
        y_test = df.loc[test_mask, 'Chyp_occur']
        
        # 训练模型
        model = elapid.MaxentModel()
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        if len(np.unique(y_test)) < 2:
            print(f"{cycle_year} test dataset only has one class, unable to calculate ROC-AUC.")
            auc = np.nan
            tss = np.nan
        else:
            y_tss = (y_pred_proba >= 0.5).astype(int)
            auc = roc_auc_score(y_test, y_pred_proba)
            tss = calculate_tss(y_test, y_tss)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, color='blue', alpha=0.3)
        
        mse = mean_squared_error(y_test, y_pred)
        feature_importance = model.permutation_importance_scores(X_train, y_train).mean(axis=1)
        
        feature_contributions.append(feature_importance)
        mse_list.append(mse)
        auc_list.append(auc)
        tss_list.append(tss)
    
    # 计算平均值
    mean_mse = np.mean(mse_list) if mse_list else np.nan
    mean_auc = np.nanmean(auc_list) if auc_list else np.nan
    mean_tss = np.nanmean(tss_list) if tss_list else np.nan
    
    # 完成ROC图
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess (AUC = 0.5)')
    plt.plot([], [], color='blue', label=f'Mean AUC: {mean_auc:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves from Multiple Experiments')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/home/yytan/ModelTest/ROCAUC{sector}_{num}_MLDaver.jpg", dpi=300)
    plt.close()
    
    # 特征贡献图
    if feature_contributions:
        mean_feature_contributions = np.mean(feature_contributions, axis=0)
        feature_contribution_df = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': mean_feature_contributions
        })
        feature_contribution_df = feature_contribution_df.sort_values(by='Contribution', ascending=True)
        
        plt.figure(figsize=(8, 6))
        plt.barh(feature_contribution_df['Feature'], feature_contribution_df['Contribution'], color='skyblue')
        for index, value in enumerate(feature_contribution_df['Contribution']):
            plt.text(value, index, f'{value:.2f}', va='center')
        plt.xlabel('Contribution')
        plt.ylabel('Feature')
        plt.title('Average Feature Contribution')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"/home/yytan/ModelTest/Contribution{sector}_{num}_MLDaver.jpg", dpi=300)
        plt.close()
    
    # GCV图
    if mse_list and auc_list and tss_list:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(cycle_years, mse_list, color='b', alpha=0.6, label=f'MSE, Mean={mean_mse:.4f}')
        ax1.set_xlabel('Cycle Year')
        ax1.set_ylabel('MSE', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        ax2 = ax1.twinx()
        ax2.plot(cycle_years, auc_list, color='r', marker='o', label=f'AUC, Mean={mean_auc:.4f}')
        ax2.plot(cycle_years, tss_list, color='lime', marker='s', label=f'TSS, Mean={mean_tss:.4f}')
        ax2.set_ylabel('AUC / TSS', color='k')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
        ax1.legend(all_lines, all_labels, loc='center right')
        plt.title('Genuine Cross Validation')
        plt.savefig(f"/home/yytan/ModelTest/GCV{sector}_{num}_MLDaver.jpg", dpi=300)
        plt.close()
    
    return mean_auc, mean_tss

def main():
    """主函数"""
    # 加载数据
    data_path = "/home/yytan/Downloads/Chyp_c1c2_mldaver.csv"
    df, feature_names = load_and_prepare_data(data_path)
    
    # 获取年份列表
    cycle_years = sorted(df['cycle_year'].unique())
    print(f"Available years: {cycle_years}")
    
    # 主循环
    for i in range(9, 5, -1):
        print(f"\nProcessing {i} features...")
        try:
            outcome = calbestComb(i, df, feature_names, cycle_years)
            outcome[0].to_csv(f"/home/yytan/Maxent/TestBestVar/ChypResultBEST{i}_MLDaver.csv", index=False)
            auc, tss = plotGCV(list(outcome[1][0]), "Chyp_MLDaver", df, cycle_years)
            print(f"Completed {i} features - AUC: {auc:.3f}, TSS: {tss:.3f}")
        except Exception as e:
            print(f"Error processing {i} features: {e}")
            continue

if __name__ == '__main__':
    main() 
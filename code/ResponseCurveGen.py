# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 19:44:21 2025

@author: Leah
"""

import elapid
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import numpy as np
from datetime import date, timedelta, datetime
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import xarray as xr
import os
from pyproj import Proj, Transformer
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, roc_curve
import itertools
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import matplotlib
import warnings

matplotlib.use('Agg')  # 使用Agg后端

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# df = pd.read_csv("/home/yytan/Downloads/Cgla_c1c2c3.csv")
df = pd.read_csv("Cgla_c1c2_mldaver.csv")


# def sum_non_nan(row):
#     if row.isna().all():
#         return np.nan
#     return row.sum()
# df["Cgla_c1c2"] = df[['Cglac1_abunm2_corr', 'Cglac2_abunm2_corr']].apply(sum_non_nan, axis=1)

df["Cgla_occur"] = df["Cgla_c1c2"].apply(lambda x:1 if x!=0 else 0)
df["cycle_year"] = df['year']
cycle_years = sorted(df['cycle_year'].unique())

feature_names = ['Sal', 'MLDaverSST', 'SST', 'SIC', 'mlotst_glor', 'MLDaverSal']


# 初始化存储所有周期响应曲线的字典
all_response_curves = {feature: [] for feature in feature_names}

# 循环每个周期年份
for cycle_year in tqdm(cycle_years, desc="Processing years"):
    # 分割训练集和测试集
    test_df = df[df['cycle_year'] == cycle_year]
    train_df = df[df['cycle_year'] != cycle_year]
    
    X_train = train_df[feature_names]
    y_train = train_df['Cgla_occur']
    X_test = test_df[feature_names]
    y_test = test_df['Cgla_occur']
    
    # 训练模型
    model = elapid.MaxentModel()
    model.fit(X_train, y_train)
    
    # 计算每个特征的响应曲线
    for feature in feature_names:
        # 创建评估范围
        min_val = df[feature].min()
        max_val = df[feature].max()
        values = np.linspace(min_val, max_val, 100)
        
        # 创建评估数据框
        eval_data = X_train.median().to_frame().T.iloc[np.zeros(100, dtype=int)].reset_index(drop=True)
        eval_data[feature] = values
        
        # 预测
        predictions = model.predict(eval_data)
        
        # 存储响应曲线
        all_response_curves[feature].append(predictions)

# 计算平均响应曲线
mean_response = {}
x_values = {}
for feature in feature_names:
    min_val = df[feature].min()
    max_val = df[feature].max()
    x_values[feature] = np.linspace(min_val, max_val, 100)
    mean_response[feature] = np.mean(all_response_curves[feature], axis=0)

# 创建2行4列的子图
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()  # 展平axes数组以便于循环

# 绘制每个特征的响应曲线
for i, feature in enumerate(feature_names):
    ax = axes[i]
    # 绘制平均响应曲线
    ax.plot(x_values[feature], mean_response[feature], 'b-', linewidth=2)
    
    # 可选：绘制所有周期响应曲线的范围
    lower = np.percentile(all_response_curves[feature], 5, axis=0)
    upper = np.percentile(all_response_curves[feature], 95, axis=0)
    ax.fill_between(x_values[feature], lower, upper, color='b', alpha=0.2)
    
    ax.set_title(feature)
    ax.set_xlabel(feature)
    ax.set_ylabel('Predicted probability')
    ax.grid(True)

# 调整布局
plt.tight_layout()
plt.savefig("E:/ECNU/20250721/responseCurve_Cgla.jpg",dpi=300)
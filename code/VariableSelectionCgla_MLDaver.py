# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:45:01 2025

@author: Yiyang Tan
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

def calculate_tss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    tss = sensitivity + specificity - 1
    return tss


df = pd.read_csv("/home/yytan/Downloads/Cgla_c1c2_mldaver.csv")

# df = df[(df['month'] >= 4) & (df['month'] <= 9)]

# def sum_non_nan(row):
#     if row.isna().all():
#         return np.nan
#     return row.sum()

# df["Cgla_c1c2"] = df[['Cglac1_abunm2_corr', 'Cglac2_abunm2_corr']].apply(sum_non_nan, axis=1)

df["Cgla_occur"] = df["Cgla_c1c2"].apply(lambda x:1 if x!=0 else 0)

df["cycle_year"] = df['year']
cycle_years = sorted(df['cycle_year'].unique())

feature_names = ['SIC', 'CHL', 'SST', 'Sal', 'nitrat', 'mlotst_glor', 'MLDaverCHL', 'MLDaverSal', 'MLDaverSST']

def calbestComb(num, df):
    results = []
    for combination in tqdm(itertools.combinations(feature_names, num)):
        remaining_features = list(combination)
        X = df[remaining_features]
        y = df['Cgla_occur']
        mse_list = []

        for cycle_year in cycle_years:
            test_df = df[df['cycle_year'] == cycle_year]
            train_df = df[df['cycle_year'] != cycle_year]
            X_train = train_df[remaining_features]
            y_train = train_df['Cgla_occur']
            X_test = test_df[remaining_features]
            y_test = test_df['Cgla_occur']
            model = elapid.MaxentModel()
            model.fit(X_train, y_train)
            if len(X_test) < 1:
                continue
            y_pred = model.predict(X_test)
            # y_pred_proba = model.predict_proba(X_test)[:, 1]

            mse = mean_squared_error(y_test, y_pred)
            mse_list.append(mse)

        mean_mse = np.mean(mse_list)
        results.append({
            'combination': combination,
            'mse': mean_mse
        })
        
    results_df = pd.DataFrame(results)
    min_mse_row = results_df.loc[results_df['mse'].idxmin()]    
    return results_df, min_mse_row


def plotGCV(feature_names, sector):
    num = len(feature_names)
    results = []
    X = df[feature_names]
    y = df['Cgla_occur']
    mse_list = []
    auc_list = []
    tss_list = []
    feature_contributions = []
        
    plt.figure(figsize=(6, 5))
    for cycle_year in cycle_years:
        test_df = df[df['cycle_year'] == cycle_year]
        train_df = df[df['cycle_year'] != cycle_year]
        X_train = train_df[feature_names]
        y_train = train_df['Cgla_occur']
        X_test = test_df[feature_names]
        y_test = test_df['Cgla_occur']
        model = elapid.MaxentModel()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
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
    mean_mse = np.mean(mse_list)
    mean_auc = np.nanmean(auc_list)
    mean_tss = np.nanmean(tss_list)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess (AUC = 0.5)')
    plt.plot([], [], color='blue', label=f'Mean AUC: {mean_auc:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves from Multiple Experiments')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/home/yytan/ModelTest/ROCAUC{sector}_{num}_MLDaver.jpg",dpi=300)
    
    mean_feature_contributions = np.mean(feature_contributions, axis=0)
    feature_contribution_df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': mean_feature_contributions
    })
    feature_contribution_df = feature_contribution_df.sort_values(by='Contribution', ascending=True)
    
    #Contribution plot
    plt.figure(figsize=(8, 6))
    plt.barh(feature_contribution_df['Feature'], feature_contribution_df['Contribution'], color='skyblue')
    for index, value in enumerate(feature_contribution_df['Contribution']):
        plt.text(value, index, f'{value:.2f}', va='center')
    plt.xlabel('Contribution')
    plt.ylabel('Feature')
    # plt.xlim(0, 15)
    plt.title('Average Feature Contribution')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"/home/yytan/ModelTest/Contribution{sector}_{num}_MLDaver.jpg",dpi=300) 
    
    # GCV plot
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
    return mean_auc, mean_tss


for i in range(9, 5, -1):
    outcome = calbestComb(i, df)
    outcome[0].to_csv(f"/home/yytan/Maxent/TestBestVar/CglaResultBEST{i}_MLDaver.csv", index=False)
    auc, tss = plotGCV(list(outcome[1][0]), "Cgla_MLDaver") 
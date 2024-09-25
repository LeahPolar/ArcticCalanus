# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:32:23 2024

@author: Yiyang Tan
"""

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import from_origin

data = pd.read_csv("D:/0-Calanus/0-SurveyData/1-abundance/Kvile1993-2016/CalanusData2017-2020.csv")
# kvile = pd.read_csv("D:\0-Calanus\0-SurveyData\1-abundance\Kvile1993-2016\CalanusData_Kvile2019_BiolBull_1993-2016.csv")

import csv

# 逐行读取文件并去除空字符
with open(r"D:\0-Calanus\0-SurveyData\1-abundance\CalanusGnH\CalanusData_Kvile2019_BiolBull_1993-2016.csv", 'r', encoding='utf-8', errors='ignore') as csvfile:
    cleaned_lines = (line.replace('\x00', '') for line in csvfile)
    reader = csv.reader(cleaned_lines)
    
    # 逐行处理 CSV 数据
    for row in reader:
        print(row)

try:
    kvile = pd.read_csv(r"D:\0-Calanus\0-SurveyData\1-abundance\CalanusGnH\CalanusData_Kvile2019_BiolBull_1993-2016.csv", encoding='utf-8', error_bad_lines=False)
    print(kvile.head())  # 显示前几行数据
except pd.errors.EmptyDataError:
    print("文件为空或包含空字符")

data2017_2020 = pd.read_csv("D:/0-Calanus/0-SurveyData/1-abundance/CalanusGnH/CalanusData2017-2020.csv")
nc_file = "D:/0-Calanus/4-Bathymetry/gebco_2024_sub_ice_topo/GEBCO_2024_sub_ice_topo.nc"
data = xr.open_dataset(nc_file, engine='netcdf4')
def get_elevation(lat, lon):
    try:
        elevation_value = data['elevation'].sel(lat=lat, lon=lon, method='nearest').values.item()
    except Exception as e:
        elevation_value = None
    return elevation_value

# 应用函数并赋值到 'depth' 列
data2017_2020['depth'] = data2017_2020.apply(lambda row: get_elevation(row['lat'], row['lon']), axis=1)

tiff_file = "D:/0-Calanus/4-Bathymetry/ibcao_v5_100m_GeoTiff/ibcao_v5_2024_100m_depth.tiff"

with rasterio.open(tiff_file) as src:
    transform = src.transform
    bathy = src.read(1)

def get_depth(lat, lon):
    row, col = ~transform * (lon, lat)
    if 0 <= row < src.height and 0 <= col < src.width:
        depth_value = bathy[int(row), int(col)]  # 读取第一个波段的深度值
    else:
        depth_value = None  # 超出范围
    return depth_value

# 应用函数并赋值到 'depth' 列
data2017_2020['depth'] = data2017_2020.apply(lambda row: get_depth(row['lat'], row['lon']), axis=1)

data2017_2020['depth'] = -data2017_2020['depth']
# 保存更新后的 DataFrame
data2017_2020.to_csv("Calanus2017-2020withdepth.csv", index=False)

data = kvile.append(data2017_2020)
data.to_csv("rowdata1997-2020.csv", index=False)

# calculate m2 abundance
data = pd.read_csv("rowdata1997-2020.csv")
for index, row in data.iterrows():
    # 遍历以 'Cglac' 和 'Chypc' 开头的列
    for col in data.columns:
        if col.endswith('_abunm3'):  # 找到每个 m3 列
            m2_col = col.replace('_abunm3', '_abunm2')  # 找到对应的 m2 列
            if pd.notna(row[col]):  # 检查 m3 列是否有值
                m2abun = (row['lowdep'] - row['updep']) * row[col]  # 计算 m2abun
                data.at[index, m2_col] = m2abun  # 将值赋给对应的 m2 列

for index, row in data.iterrows():
    # 计算 Cgla_abunm3 和 Cgla_abunm2
    if pd.isna(row['Cgla_abunm3']):
        # 找到所有以 'Cgla' 开头且以 '_abunm3' 结尾的列
        m3_sum = row.filter(like='Cgla').filter(like='_abunm3').sum()
        data.at[index, 'Cgla_abunm3'] = m3_sum  # 赋值

    if pd.isna(row['Cgla_abunm2']):
        # 找到所有以 'Cgla' 开头且以 '_abunm2' 结尾的列
        m2_sum = row.filter(like='Cgla').filter(like='_abunm2').sum()
        data.at[index, 'Cgla_abunm2'] = m2_sum  # 赋值

    # 对 Chyp 进行同样的处理
    if pd.isna(row['Chyp_abunm3']):
        m3_sum = row.filter(like='Chyp').filter(like='_abunm3').sum()
        data.at[index, 'Chyp_abunm3'] = m3_sum  # 赋值

    if pd.isna(row['Chyp_abunm2']):
        m2_sum = row.filter(like='Chyp').filter(like='_abunm2').sum()
        data.at[index, 'Chyp_abunm2'] = m2_sum  # 赋值

data.to_csv("data1997-2020filled.csv", index=False)

# 删除以 'm3' 结尾的列
m3_columns = [col for col in data.columns if col.endswith('m3')]
datam2 = data.drop(columns=m3_columns)

def process_continuous_depth(group):
    if len(group) == 1:
        return group
    
    sorted_group = group.sort_values(by='updep')
    new_rows = []
    
    for i in range(len(sorted_group) - 1):
        current_row = sorted_group.iloc[i]
        next_row = sorted_group.iloc[i + 1]
        
        if current_row['lowdep'] == next_row['updep']:
            # 深度连续，直接累加
            continue
        elif current_row['lowdep'] > next_row['updep']:
            # 插入新的行
            new_row = current_row.copy()
            new_row['updep'] = current_row['lowdep']
            new_row['lowdep'] = next_row['updep']
            
            # 计算m2结尾列的平均值
            m2_columns = [col for col in sorted_group.columns if col.endswith('m2')]
            for col in m2_columns:
                new_row[col] = (current_row[col] + next_row[col]) / 2
            
            new_rows.append(new_row)
        elif current_row['lowdep'] < next_row['updep']:
            # 调整m2结尾的列
            overlap = next_row['updep'] - current_row['lowdep']
            factor1 = 1 - (overlap / 2) / (current_row['lowdep'] - current_row['updep'])
            factor2 = 1 - (overlap / 2) / (next_row['lowdep'] - next_row['updep'])
            
            m2_columns = [col for col in sorted_group.columns if col.endswith('m2')]
            for col in m2_columns:
                current_row[col] *= factor1
                next_row[col] *= factor2
    
    # 将新行添加到原始组中
    if new_rows:
        sorted_group = pd.concat([sorted_group, pd.DataFrame(new_rows)], ignore_index=True)
        sorted_group = sorted_group.sort_values(by='updep')
    
    return sorted_group

grouped = datam2.groupby(['year', 'month', 'day', 'julday', 'lat', 'lon'])
processed_groups = {name: process_continuous_depth(group) for name, group in grouped}

def merge_continuous_group(group):
    if len(group) == 1:
        return group.iloc[0]
    
    # 对以 'm2' 结尾的列进行求和
    sum_columns = [col for col in group.columns if col.endswith('m2')]
    summed_data = group[sum_columns].sum()
    
    # 保留其他列的第一行数据
    other_columns = [col for col in group.columns if not col.endswith('m2')]
    other_data = group[other_columns].iloc[0]
    
    # 更新 updep 和 lowdep
    other_data['updep'] = group['updep'].iloc[0]
    other_data['lowdep'] = group['lowdep'].iloc[-1]
    
    # 合并结果
    result = pd.concat([other_data, summed_data])
    return result

merged_data = pd.DataFrame([merge_continuous_group(group) for group in processed_groups.values()])

merged_data.to_csv("standardized1997-2020.csv", index=False)

# calculate total abundance
bodysize = pd.read_csv("D:/0-Calanus/0-SurveyData/1-abundance/CalanusGnH/CopepodWidths_and_NetMeshSizes_KK.csv")

def calculate_new_column(row, original_column, new_table):
    # 获取当前行的 mesh 值
    mesh_value = row['mesh']
    
    # 获取对应生命阶段的体长
    species = species = original_column[:4]   # 获取物种名称（Cgla 或 Chyp）
    stage = original_column.split('_')[0][4:]  # 获取生命阶段（c1, c2, c3, ...）
    
    if species == 'Cgla':
        length_value = new_table.loc[new_table['Stage'] == stage, 'Cgla'].values[0]
    elif species == 'Chyp':
        length_value = new_table.loc[new_table['Stage'] == stage, 'Chyp'].values[0]
    
    # 计算 R
    R = length_value / mesh_value
    # 计算新列的值
    valid = 1 / (1 + np.exp(-8.9 * (R - 1)))
    
    # 如果新列的值小于 25%，则返回空值
    if valid < 0.25:
        return np.nan
    
    # 计算新列的值
    new_value = row[original_column] * (1 + np.exp(-8.9 * (R - 1)))
    
    return new_value

# 获取所有以 'm2' 结尾的列
m2_columns = [col for col in merged_data.columns if col.endswith('m2') and col not in ['Cgla_abunm2', 'Chyp_abunm2']]

# 对每一列进行计算并生成新列
for col in m2_columns:
    new_column_name = col + '_corr'
    merged_data[new_column_name] = merged_data.apply(lambda row: calculate_new_column(row, col, bodysize), axis=1)

# 获取所有以 'Cgla' 开始并以 'abunm2_corr' 结尾的列
cgla_corr_columns = [col for col in merged_data.columns if col.startswith('Cgla') and col.endswith('abunm2_corr')]

# 获取所有以 'Chyp' 开始并以 'abunm2_corr' 结尾的列
chyp_corr_columns = [col for col in merged_data.columns if col.startswith('Chyp') and col.endswith('abunm2_corr')]

# 计算 Cgla_abunm2_corr 和 Chyp_abunm2_corr
merged_data['Cgla_abunm2_corr'] = merged_data[cgla_corr_columns].sum(axis=1)
merged_data['Chyp_abunm2_corr'] = merged_data[chyp_corr_columns].sum(axis=1)

merged_data.to_csv("final_standardized_m2abun1993-2020.csv")
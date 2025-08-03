# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:18:24 2024

@author: Yiyang Tan
修正版本：根据正确的TOA/TOR计算逻辑
TOA: 从当年SIE最小日期向后遍历到下一年SIE最小日期，寻找第一次海冰浓度>0.15的日期
TOR: 从当年SIE最小日期向前遍历到上一年SIE最小日期（或当年1月1日），寻找第一次海冰浓度>0.15的日期
"""

import os
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import netCDF4 as nc
import rasterio
from rasterio.transform import from_origin

# 参数
sie_csv = '/home/yytan/Downloads/SIE_min_date.csv'
nc_dir = '/data/oceandata/GLORYS2v4Combine/'
output_dir = '/home/yytan/Variables/Seaice/'
os.makedirs(output_dir, exist_ok=True)

# 读取SIE最小日期
sie_df = pd.read_csv(sie_csv, index_col=0, parse_dates=['min-date'])

# 年份范围
start_year = 1993
end_year = 2022

# 经纬度信息
transform = from_origin(-180, 40, 0.25, 0.25)

def get_julian_day(date):
    return date.timetuple().tm_yday

for year in range(start_year, end_year + 1):
    print(f'处理年份: {year}')
    nc_file = f'{nc_dir}GLORYS2v4_{year}.nc'
    if not os.path.exists(nc_file):
        print(f"文件不存在: {nc_file}")
        continue
    
    ds = nc.Dataset(nc_file)
    sic = ds.variables['sic'][:]  # (天, lat, lon)
    base_date = datetime.datetime(year, 1, 1)
    days_in_year = sic.shape[0]

    # 读取min-date
    min_date_this = sie_df.loc[year, 'min-date'] if year in sie_df.index else None
    min_date_next = sie_df.loc[year + 1, 'min-date'] if (year + 1) in sie_df.index else None
    min_date_prev = sie_df.loc[year - 1, 'min-date'] if (year - 1) in sie_df.index else None

    print(f"  SIE最小日期: {min_date_this}")
    print(f"  下一年SIE最小日期: {min_date_next}")
    print(f"  上一年SIE最小日期: {min_date_prev}")

    # TOA计算 - 修正版本
    toa_matrix = np.full(sic.shape[1:], np.nan)
    if min_date_this is not None and min_date_next is not None:
        # TOA: 从当年SIE最小日期向后遍历到下一年SIE最小日期
        start_idx = (min_date_this - base_date).days
        end_idx = (min_date_next - base_date).days
        
        # 确保索引在有效范围内
        start_idx = max(0, min(start_idx, days_in_year - 1))
        end_idx = max(0, min(end_idx, days_in_year))
        
        print(f"  TOA搜索范围: {start_idx} 到 {end_idx} (从{min_date_this.strftime('%Y-%m-%d')}到{min_date_next.strftime('%Y-%m-%d')})")
        
        for lat in tqdm(range(sic.shape[1]), desc=f"TOA计算 - 纬度", leave=False):
            for lon in range(sic.shape[2]):
                found_toa = False
                for i in range(start_idx, end_idx):
                    if i < days_in_year and not np.isnan(sic[i, lat, lon]) and sic[i, lat, lon] > 0.15:
                        # 计算儒略日：目标日期减去当年1月1日再加1
                        target_date = base_date + datetime.timedelta(days=i)
                        julian_day = get_julian_day(target_date)
                        toa_matrix[lat, lon] = julian_day
                        found_toa = True
                        break
                
                # 如果没有找到TOA，保持NaN值
                if not found_toa:
                    toa_matrix[lat, lon] = np.nan

    # TOR计算 - 修正版本
    tor_matrix = np.full(sic.shape[1:], np.nan)
    if min_date_this is not None:
        # TOR: 从当年SIE最小日期向前遍历
        start_idx = (min_date_this - base_date).days
        
        # 确定结束索引
        if year == 1993:
            # 1993年：遍历到1993年1月1日
            end_idx = 0
        else:
            # 其他年份：遍历到上一年SIE最小日期
            if min_date_prev is not None:
                end_idx = (min_date_prev - base_date).days
            else:
                end_idx = 0
        
        # 确保索引在有效范围内
        start_idx = max(0, min(start_idx, days_in_year - 1))
        end_idx = max(0, min(end_idx, days_in_year))
        
        if year == 1993:
            print(f"  TOR搜索范围: {start_idx} 到 {end_idx} (从{min_date_this.strftime('%Y-%m-%d')}到1993-01-01)")
        else:
            print(f"  TOR搜索范围: {start_idx} 到 {end_idx} (从{min_date_this.strftime('%Y-%m-%d')}到{min_date_prev.strftime('%Y-%m-%d')})")
        
        for lat in tqdm(range(sic.shape[1]), desc=f"TOR计算 - 纬度", leave=False):
            for lon in range(sic.shape[2]):
                found_tor = False
                for i in range(start_idx, end_idx, -1):  # 反向遍历
                    if i < days_in_year and not np.isnan(sic[i, lat, lon]) and sic[i, lat, lon] > 0.15:
                        # 计算儒略日：目标日期减去当年1月1日再加1
                        target_date = base_date + datetime.timedelta(days=i)
                        julian_day = get_julian_day(target_date)
                        tor_matrix[lat, lon] = julian_day
                        found_tor = True
                        break
                
                # 如果没有找到TOR，保持NaN值
                if not found_tor:
                    tor_matrix[lat, lon] = np.nan

    # 统计结果
    toa_valid = np.sum(~np.isnan(toa_matrix))
    tor_valid = np.sum(~np.isnan(tor_matrix))
    total_pixels = toa_matrix.size
    
    print(f"  TOA有效像素: {toa_valid}/{total_pixels} ({toa_valid/total_pixels*100:.1f}%)")
    print(f"  TOR有效像素: {tor_valid}/{total_pixels} ({tor_valid/total_pixels*100:.1f}%)")
    
    if toa_valid > 0:
        print(f"  TOA范围: {np.nanmin(toa_matrix):.1f} - {np.nanmax(toa_matrix):.1f}")
    if tor_valid > 0:
        print(f"  TOR范围: {np.nanmin(tor_matrix):.1f} - {np.nanmax(tor_matrix):.1f}")

    # 保存TOA
    toa_matrix = np.flipud(toa_matrix)
    with rasterio.open(
        os.path.join(output_dir, f"TOA_{year}.tif"),
        'w',
        driver='GTiff',
        height=toa_matrix.shape[0],
        width=toa_matrix.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(toa_matrix.astype(np.float32), 1)

    # 保存TOR
    tor_matrix = np.flipud(tor_matrix)
    with rasterio.open(
        os.path.join(output_dir, f"TOR_{year}.tif"),
        'w',
        driver='GTiff',
        height=tor_matrix.shape[0],
        width=tor_matrix.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(tor_matrix.astype(np.float32), 1)

    ds.close()

print("所有年份处理完成！") 
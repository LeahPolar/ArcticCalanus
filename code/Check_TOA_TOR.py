# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:18:24 2024

@author: Yiyang Tan
"""

import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import netCDF4 as nc
import datetime

def check_seaice_files(year):
    """
    检查指定年份的TOA和TOR文件
    
    Parameters:
    -----------
    year : int
        要检查的年份
    """
    toa_file = f'/home/yytan/Variables/Seaice/TOA_{year}.tif'
    tor_file = f'/home/yytan/Variables/Seaice/TOR_{year}.tif'
    
    print(f"\n{'='*50}")
    print(f"检查年份: {year}")
    print(f"{'='*50}")
    
    # 检查文件是否存在
    if not os.path.exists(toa_file):
        print(f"TOA文件不存在: {toa_file}")
        return
    if not os.path.exists(tor_file):
        print(f"TOR文件不存在: {tor_file}")
        return
    
    # 读取TOA数据
    try:
        with rasterio.open(toa_file) as src:
            toa_data = src.read(1)
            print(f"TOA文件信息:")
            print(f"  Shape: {toa_data.shape}")
            print(f"  Transform: {src.transform}")
            print(f"  Bounds: {src.bounds}")
    except Exception as e:
        print(f"读取TOA文件错误: {e}")
        return
    
    # 读取TOR数据
    try:
        with rasterio.open(tor_file) as src:
            tor_data = src.read(1)
            print(f"TOR文件信息:")
            print(f"  Shape: {tor_data.shape}")
            print(f"  Transform: {src.transform}")
            print(f"  Bounds: {src.bounds}")
    except Exception as e:
        print(f"读取TOR文件错误: {e}")
        return
    
    # 比较两个文件
    print(f"\n数据比较:")
    print(f"TOA数据范围: {np.nanmin(toa_data):.1f} 到 {np.nanmax(toa_data):.1f}")
    print(f"TOR数据范围: {np.nanmin(tor_data):.1f} 到 {np.nanmax(tor_data):.1f}")
    
    print(f"TOA非NaN值数量: {np.sum(~np.isnan(toa_data))}")
    print(f"TOR非NaN值数量: {np.sum(~np.isnan(tor_data))}")
    
    # 检查是否完全相同
    if np.array_equal(toa_data, tor_data):
        print("⚠️  警告: TOA和TOR文件完全相同！")
    else:
        print("✅ TOA和TOR文件不同")
    
    # 检查相关性
    valid_mask = ~(np.isnan(toa_data) | np.isnan(tor_data))
    if np.sum(valid_mask) > 0:
        correlation = np.corrcoef(toa_data[valid_mask], tor_data[valid_mask])[0, 1]
        print(f"TOA和TOR相关性: {correlation:.3f}")
    
    # 显示一些随机样本
    print(f"\n随机样本比较 (前10个非NaN值):")
    valid_indices = np.where(valid_mask)
    if len(valid_indices[0]) > 0:
        sample_size = min(10, len(valid_indices[0]))
        random_indices = np.random.choice(len(valid_indices[0]), sample_size, replace=False)
        
        for i in random_indices:
            lat_idx, lon_idx = valid_indices[0][i], valid_indices[1][i]
            toa_val = toa_data[lat_idx, lon_idx]
            tor_val = tor_data[lat_idx, lon_idx]
            print(f"  位置({lat_idx}, {lon_idx}): TOA={toa_val:.1f}, TOR={tor_val:.1f}")
    
    # 显示数据分布
    print(f"\n数据分布:")
    toa_valid = toa_data[~np.isnan(toa_data)]
    tor_valid = tor_data[~np.isnan(tor_data)]
    
    if len(toa_valid) > 0:
        print(f"TOA - 均值: {np.mean(toa_valid):.1f}, 中位数: {np.median(toa_valid):.1f}, 标准差: {np.std(toa_valid):.1f}")
    if len(tor_valid) > 0:
        print(f"TOR - 均值: {np.mean(tor_valid):.1f}, 中位数: {np.median(tor_valid):.1f}, 标准差: {np.std(tor_valid):.1f}")

def check_sie_data():
    """
    检查SIE最小日期数据
    """
    sie_csv = '/home/yytan/Downloads/SIE_min_date.csv'
    if not os.path.exists(sie_csv):
        print(f"SIE文件不存在: {sie_csv}")
        return
    
    print(f"\n{'='*50}")
    print("检查SIE最小日期数据")
    print(f"{'='*50}")
    
    sie_df = pd.read_csv(sie_csv, index_col=0, parse_dates=['min-date'])
    print(f"SIE数据形状: {sie_df.shape}")
    print(f"年份范围: {sie_df.index.min()} - {sie_df.index.max()}")
    
    print(f"\nSIE最小日期:")
    for year in sie_df.index:
        min_date = sie_df.loc[year, 'min-date']
        print(f"  {year}: {min_date.strftime('%Y-%m-%d')} (儒略日: {min_date.timetuple().tm_yday})")
    
    # 检查相邻年份的日期差异
    print(f"\n相邻年份日期差异:")
    for i in range(len(sie_df.index) - 1):
        year1 = sie_df.index[i]
        year2 = sie_df.index[i + 1]
        date1 = sie_df.loc[year1, 'min-date']
        date2 = sie_df.loc[year2, 'min-date']
        diff_days = (date2 - date1).days
        print(f"  {year1}-{year2}: {diff_days} 天")

def check_original_sic_data(year):
    """
    检查原始海冰浓度数据
    """
    nc_file = f'/data/oceandata/GLORYS2v4Combine/GLORYS2v4_{year}.nc'
    if not os.path.exists(nc_file):
        print(f"GLORYS2v4文件不存在: {nc_file}")
        return
    
    print(f"\n{'='*50}")
    print(f"检查原始海冰浓度数据 - 年份: {year}")
    print(f"{'='*50}")
    
    try:
        ds = nc.Dataset(nc_file)
        sic = ds.variables['sic'][:]  # (天, lat, lon)
        print(f"海冰浓度数据形状: {sic.shape}")
        print(f"数据范围: {np.nanmin(sic):.3f} 到 {np.nanmax(sic):.3f}")
        
        # 检查海冰浓度 > 0.15 的像素数量随时间变化
        print(f"\n海冰浓度 > 0.15 的像素数量随时间变化:")
        for i in range(0, sic.shape[0], 30):  # 每30天检查一次
            day_count = np.sum(sic[i, :, :] > 0.15)
            date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=i)
            print(f"  {date.strftime('%Y-%m-%d')}: {day_count} 像素")
        
        ds.close()
        
    except Exception as e:
        print(f"读取GLORYS2v4文件错误: {e}")

def check_multiple_years(start_year=1993, end_year=2022, sample_size=5):
    """
    检查多个年份的文件
    
    Parameters:
    -----------
    start_year : int
        开始年份
    end_year : int
        结束年份
    sample_size : int
        随机检查的年份数量
    """
    print(f"检查{start_year}-{end_year}年间的{sample_size}个随机年份")
    
    # 随机选择年份
    years = np.random.choice(range(start_year, end_year + 1), sample_size, replace=False)
    
    for year in years:
        check_seaice_files(year)

def check_specific_locations(year, lat_lons):
    """
    检查特定位置的TOA和TOR值
    
    Parameters:
    -----------
    year : int
        年份
    lat_lons : list of tuples
        要检查的(纬度, 经度)坐标列表
    """
    toa_file = f'/home/yytan/Variables/Seaice/TOA_{year}.tif'
    tor_file = f'/home/yytan/Variables/Seaice/TOR_{year}.tif'
    
    if not os.path.exists(toa_file) or not os.path.exists(tor_file):
        print(f"文件不存在: {toa_file} 或 {tor_file}")
        return
    
    with rasterio.open(toa_file) as src:
        toa_data = src.read(1)
    with rasterio.open(tor_file) as src:
        tor_data = src.read(1)
    
    print(f"\n检查年份{year}的特定位置:")
    print(f"{'='*40}")
    
    for lat, lon in lat_lons:
        # 计算像素索引
        lon_idx = int((lon - (-180)) / 0.25)
        lat_range = np.arange(40, -10.1, -0.25)
        lat_idx = np.argmin(np.abs(lat_range - lat))
        
        if 0 <= lat_idx < len(lat_range) and 0 <= lon_idx < 1440:
            toa_val = toa_data[lat_idx, lon_idx]
            tor_val = tor_data[lat_idx, lon_idx]
            print(f"位置({lat:.2f}, {lon:.2f}) -> 像素({lat_idx}, {lon_idx}): TOA={toa_val}, TOR={tor_val}")
        else:
            print(f"位置({lat:.2f}, {lon:.2f}) -> 超出边界")

def main():
    """
    主函数
    """
    print("海冰TOA/TOR文件检查工具")
    print("=" * 60)
    
    # 检查SIE数据
    check_sie_data()
    
    # 检查原始海冰浓度数据
    check_original_sic_data(2020)
    
    # 检查几个随机年份
    check_multiple_years(1993, 2022, 5)
    
    # 检查特定位置（北极区域的一些典型坐标）
    test_locations = [
        (75.0, -45.0),   # 格陵兰海
        (80.0, 30.0),    # 巴伦支海
        (70.0, -150.0),  # 波弗特海
        (85.0, 0.0),     # 北极中心
        (60.0, -90.0),   # 哈德逊湾
    ]
    
    check_specific_locations(2020, test_locations)
    
    print(f"\n{'='*60}")
    print("检查完成！")
    print("如果TOA和TOR值相同，可能的原因：")
    print("1. SIE最小日期数据有问题")
    print("2. 海冰浓度阈值设置不当")
    print("3. 计算逻辑有误")
    print("4. 数据保存时出现问题")

if __name__ == '__main__':
    main() 
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:18:24 2024

@author: Yiyang Tan
匹配每条记录对应的经纬度和时间过去60天的平均海冰密集度
"""

import os
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import netCDF4 as nc

def calculate_p60dmSIC_for_record(row, nc_dir):
    """
    计算单条记录过去60天的平均海冰密集度
    """
    lat = row['lat']
    lon = row['lon']
    year = int(row['year'])
    month = int(row['month'])
    day = int(row['day'])
    
    # 计算目标日期
    target_date = datetime.datetime(year, month, day)
    
    # 计算过去60天的日期范围
    end_date = target_date - datetime.timedelta(days=1)  # 不包括当天
    start_date = end_date - datetime.timedelta(days=59)  # 过去60天
    
    sic_values = []
    
    # 获取需要读取的年份
    years_needed = set()
    current_date = start_date
    while current_date <= end_date:
        years_needed.add(current_date.year)
        current_date += datetime.timedelta(days=1)
    
    # 读取所有需要的年份数据
    datasets = {}
    for year in years_needed:
        nc_file = f'{nc_dir}GLORYS2v4_{year}.nc'
        if os.path.exists(nc_file):
            try:
                datasets[year] = nc.Dataset(nc_file)
            except Exception as e:
                print(f"读取文件 {nc_file} 时出错: {e}")
    
    # 遍历过去60天的每一天
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        
        if year in datasets:
            try:
                ds = datasets[year]
                
                # 计算该日期在年份中的天数索引
                base_date = datetime.datetime(year, 1, 1)
                day_idx = (current_date - base_date).days
                
                # 检查索引是否有效
                if 0 <= day_idx < ds.dimensions['time'].size:
                    # 获取经纬度信息 - slat和slon是2D变量
                    slat = ds.variables['slat'][:]  # (201, 1440)
                    slon = ds.variables['slon'][:]  # (201, 1440)
                    
                    # 找到最接近的经纬度位置
                    # 计算每个网格点到目标点的距离
                    distances = np.sqrt((slat - lat)**2 + (slon - lon)**2)
                    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                    lat_idx, lon_idx = min_idx
                    
                    # 检查索引是否在有效范围内
                    if 0 <= lat_idx < slat.shape[0] and 0 <= lon_idx < slat.shape[1]:
                        # 直接通过索引读取SIC数据，类似于MaxentPlot-supp.py的方式
                        sic = ds.variables['sic'][day_idx, :, :]
                        
                        # 获取指定位置的值
                        sic_val = sic[lat_idx, lon_idx]
                        
                        # 检查是否为有效数据
                        if not np.isnan(sic_val) and not np.ma.is_masked(sic_val):
                            sic_values.append(float(sic_val))
                    else:
                        print(f"经纬度索引超出范围: lat_idx={lat_idx}, lon_idx={lon_idx}, lat={lat}, lon={lon}")
                        print(f"经纬度范围: lat[{slat.min():.2f}, {slat.max():.2f}], lon[{slon.min():.2f}, {slon.max():.2f}]")
                        
            except Exception as e:
                print(f"处理日期 {current_date} 时出错: {e}")
        
        current_date += datetime.timedelta(days=1)
    
    # 关闭所有数据集
    for ds in datasets.values():
        ds.close()
    
    # 计算平均值
    if sic_values:
        return np.mean(sic_values)
    else:
        return np.nan

def match_p60dmSIC_for_file(input_file, output_file, nc_dir):
    """
    为单个文件匹配过去60天的平均海冰密集度
    """
    print(f"处理文件: {input_file}")
    
    # 读取数据
    df = pd.read_csv(input_file)
    print(f"  总记录数: {len(df)}")
    
    # 确保年月日为整数
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    
    # 计算过去60天的平均海冰密集度
    p60dmSIC_values = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="计算p60dmSIC"):
        p60dmSIC = calculate_p60dmSIC_for_record(row, nc_dir)
        p60dmSIC_values.append(p60dmSIC)
    
    # 添加新列
    df['p60dmSIC'] = p60dmSIC_values
    
    # 统计结果
    valid_count = np.sum(~np.isnan(df['p60dmSIC']))
    print(f"  有效匹配数: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")
    
    if valid_count > 0:
        print(f"  p60dmSIC范围: {np.nanmin(df['p60dmSIC']):.3f} - {np.nanmax(df['p60dmSIC']):.3f}")
        print(f"  p60dmSIC平均值: {np.nanmean(df['p60dmSIC']):.3f}")
    
    # 保存结果
    df.to_csv(output_file, index=False)
    print(f"  结果已保存到: {output_file}")
    
    return df

def main():
    """
    主函数
    """
    # 参数设置
    nc_dir = '/data/oceandata/GLORYS2v4Combine/'
    
    # 输入文件
    cgla_input = '/home/yytan/Downloads/Cgla_c1c2_mldaver.csv'
    chyp_input = '/home/yytan/Downloads/Chyp_c1c2_mldaver.csv'
    
    # 输出文件
    cgla_output = '/home/yytan/Downloads/Cgla_c1c2_p60dmSIC.csv'
    chyp_output = '/home/yytan/Downloads/Chyp_c1c2_p60dmSIC.csv'
    
    # 检查输入文件是否存在
    if not os.path.exists(cgla_input):
        print(f"错误：输入文件不存在: {cgla_input}")
        return
    
    if not os.path.exists(chyp_input):
        print(f"错误：输入文件不存在: {chyp_input}")
        return
    
    # 处理Cgla文件
    print("=" * 50)
    print("处理Cgla文件")
    print("=" * 50)
    cgla_df = match_p60dmSIC_for_file(cgla_input, cgla_output, nc_dir)
    
    # 处理Chyp文件
    print("\n" + "=" * 50)
    print("处理Chyp文件")
    print("=" * 50)
    chyp_df = match_p60dmSIC_for_file(chyp_input, chyp_output, nc_dir)
    
    print("\n" + "=" * 50)
    print("所有文件处理完成！")
    print("=" * 50)
    print(f"Cgla结果: {cgla_output}")
    print(f"Chyp结果: {chyp_output}")

if __name__ == "__main__":
    main() 
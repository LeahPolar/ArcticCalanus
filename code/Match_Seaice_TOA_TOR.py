# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:18:24 2024

@author: Yiyang Tan
"""

import pandas as pd
import numpy as np
import rasterio
from tqdm import tqdm
import os
import datetime

def debug_seaice_file(file_path):
    """调试函数：检查海冰文件的坐标信息"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        with rasterio.open(file_path) as src:
            print(f"\nFile: {os.path.basename(file_path)}")
            print(f"Shape: {src.shape}")
            print(f"Transform: {src.transform}")
            print(f"Bounds: {src.bounds}")
            print(f"CRS: {src.crs}")
            
            # 读取数据并显示一些统计信息
            data = src.read(1)
            print(f"Data shape: {data.shape}")
            print(f"Data range: {np.nanmin(data):.1f} to {np.nanmax(data):.1f}")
            print(f"Non-NaN values: {np.sum(~np.isnan(data))}")
            
            # 显示一些示例值
            print("示例像素值（前5x5区域）：")
            print(data[:5, :5])
            
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

def match_seaice_data_for_file(input_file, output_file):
    """
    为单个文件匹配海冰数据
    
    Parameters:
    -----------
    input_file : str
        输入CSV文件路径
    output_file : str
        输出CSV文件路径
    """
    print(f"Processing {os.path.basename(input_file)}...")
    
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 添加新的列
    df['TOA'] = np.nan
    df['TOR'] = np.nan
    
    # 获取唯一的年份列表
    years = df['year'].unique()
    print(f"Found years: {sorted(years)}")
    
    # 为每个年份加载对应的TOA和TOR文件
    seaice_data = {}
    for year in years:
        year = int(year)
        toa_file = f'/home/yytan/Variables/Seaice/TOA_{year}.tif'
        tor_file = f'/home/yytan/Variables/Seaice/TOR_{year}.tif'
        
        # 调试第一个年份的文件
        if year == years[0]:
            print(f"\n调试第一个年份（{year}）的文件：")
            debug_seaice_file(toa_file)
            debug_seaice_file(tor_file)
        
        # 加载TOA数据
        if os.path.exists(toa_file):
            try:
                with rasterio.open(toa_file) as src:
                    toa_data = src.read(1)
                    seaice_data[f'TOA_{year}'] = toa_data
                print(f"Loaded TOA data for year {year}")
            except Exception as e:
                print(f"Error loading TOA file for year {year}: {e}")
                seaice_data[f'TOA_{year}'] = None
        else:
            print(f"TOA file not found for year {year}: {toa_file}")
            seaice_data[f'TOA_{year}'] = None
        
        # 加载TOR数据
        if os.path.exists(tor_file):
            try:
                with rasterio.open(tor_file) as src:
                    tor_data = src.read(1)
                    seaice_data[f'TOR_{year}'] = tor_data
                print(f"Loaded TOR data for year {year}")
            except Exception as e:
                print(f"Error loading TOR file for year {year}: {e}")
                seaice_data[f'TOR_{year}'] = None
        else:
            print(f"TOR file not found for year {year}: {tor_file}")
            seaice_data[f'TOR_{year}'] = None
    
    # 遍历每一行进行匹配
    matched_count = 0
    for idx in tqdm(df.index, desc=f"Matching seaice data for {os.path.basename(input_file)}"):
        row = df.iloc[idx]
        
        # 获取坐标和日期信息
        lat, lon = row['lat'], row['lon']
        year, month, day = int(row['year']), int(row['month']), int(row['day'])
        
        # 计算像素索引
        # 根据transform = from_origin(-180, 40, 0.25, 0.25)计算
        # 经度从-180开始，纬度从40开始，分辨率0.25度
        # 注意：TOA和TOR文件在保存时使用了np.flipud()，所以纬度轴被翻转了
        lon_idx = int((lon - (-180)) / 0.25)
        
        # 由于使用了np.flipud()，纬度范围从-10到40度，翻转后变成40到-10度
        # 所以对于给定的lat，我们需要找到翻转后的索引
        lat_range = np.arange(40, -10.1, -0.25)  # 从40度到-10度，步长0.25
        lat_idx = np.argmin(np.abs(lat_range - lat))
        
        # 边界检查
        if 0 <= lat_idx < len(lat_range) and 0 <= lon_idx < 1440:
            # 获取对应年份的数据
            toa_key = f'TOA_{year}'
            tor_key = f'TOR_{year}'
            
            # 匹配TOA数据
            if toa_key in seaice_data and seaice_data[toa_key] is not None:
                try:
                    toa_value = seaice_data[toa_key][lat_idx, lon_idx]
                    if not np.isnan(toa_value):
                        df.at[idx, 'TOA'] = toa_value
                        matched_count += 1
                except IndexError:
                    print(f"Index error for TOA at lat_idx={lat_idx}, lon_idx={lon_idx}")
                except Exception as e:
                    print(f"Error reading TOA value: {e}")
            
            # 匹配TOR数据
            if tor_key in seaice_data and seaice_data[tor_key] is not None:
                try:
                    tor_value = seaice_data[tor_key][lat_idx, lon_idx]
                    if not np.isnan(tor_value):
                        df.at[idx, 'TOR'] = tor_value
                except IndexError:
                    print(f"Index error for TOR at lat_idx={lat_idx}, lon_idx={lon_idx}")
                except Exception as e:
                    print(f"Error reading TOR value: {e}")
        else:
            if idx < 10:  # 只打印前10个超出边界的坐标作为示例
                print(f"Coordinates out of bounds: lat={lat}, lon={lon}, lat_idx={lat_idx}, lon_idx={lon_idx}")
    
    # 保存结果
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # 统计匹配结果
    toa_matched = df['TOA'].notna().sum()
    tor_matched = df['TOR'].notna().sum()
    total_rows = len(df)
    
    print(f"TOA matched: {toa_matched}/{total_rows} ({toa_matched/total_rows*100:.1f}%)")
    print(f"TOR matched: {tor_matched}/{total_rows} ({tor_matched/total_rows*100:.1f}%)")
    print(f"Total successful matches: {matched_count}")
    
    # 显示一些示例匹配结果
    print("\n示例匹配结果（前5行）：")
    sample_df = df.head()
    for idx, row in sample_df.iterrows():
        print(f"Row {idx}: lat={row['lat']:.2f}, lon={row['lon']:.2f}, TOA={row['TOA']}, TOR={row['TOR']}")
    
    return df

def main():
    """
    主函数：处理Cgla和Chyp两个文件
    """
    print("=" * 60)
    print("开始匹配海冰消长数据（TOA和TOR）")
    print("=" * 60)
    
    # 定义输入和输出文件路径
    input_files = [
        '/home/yytan/Downloads/Cgla_c1c2_mldaver.csv',
        '/home/yytan/Downloads/Chyp_c1c2_mldaver.csv'
    ]
    
    output_files = [
        '/home/yytan/Downloads/Cgla_c1c2_mldaver_seaice.csv',
        '/home/yytan/Downloads/Chyp_c1c2_mldaver_seaice.csv'
    ]
    
    # 处理每个文件
    for input_file, output_file in zip(input_files, output_files):
        print(f"\n{'='*40}")
        print(f"Processing: {os.path.basename(input_file)}")
        print(f"{'='*40}")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            continue
        
        try:
            # 匹配海冰数据
            df_result = match_seaice_data_for_file(input_file, output_file)
            
            # 显示数据预览
            print(f"\nData preview for {os.path.basename(output_file)}:")
            print(f"Total rows: {len(df_result)}")
            print(f"Columns: {list(df_result.columns)}")
            print(f"TOA range: {df_result['TOA'].min():.1f} to {df_result['TOA'].max():.1f}")
            print(f"TOR range: {df_result['TOR'].min():.1f} to {df_result['TOR'].max():.1f}")
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("海冰消长数据匹配完成！")
    print("=" * 60)
    print("输出文件：")
    for output_file in output_files:
        print(f"  - {output_file}")

if __name__ == '__main__':
    main() 
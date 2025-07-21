import os
import pandas as pd
import numpy as np
import rasterio
from datetime import datetime
from tqdm import tqdm

# 输入文件路径
INPUT_FILES = [
    '/home/yytan/Downloads/Cgla_c1c2.csv',
    '/home/yytan/Downloads/Chyp_c1c2.csv'
]

# MLD平均文件路径模板
MLD_BASE_PATH = '/home/yytan/Variables/'
VARIABLES = {
    'MLDaverCHL': 'CHL',
    'MLDaverSal': 'Sal', 
    'MLDaverSST': 'SST'
}

def get_pixel_value_from_tif(tif_path, lat, lon):
    """从tif文件中读取指定经纬度坐标的像素值"""
    try:
        with rasterio.open(tif_path) as src:
            # 将经纬度转换为像素坐标
            row, col = src.index(lon, lat)
            # 读取像素值
            value = src.read(1)[row, col]
            return float(value) if not np.isnan(value) else np.nan
    except Exception as e:
        print(f"Error reading {tif_path}: {e}")
        return np.nan

def match_mld_average_for_file(input_file):
    """为单个文件匹配MLD平均值"""
    print(f"Processing {input_file}")
    
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 添加新列
    for col_name, var_name in VARIABLES.items():
        df[col_name] = np.nan
    
    # 遍历每一行
    for idx in tqdm(df.index, desc=f"Processing {os.path.basename(input_file)}"):
        row = df.iloc[idx]
        lat, lon = row['lat'], row['lon']
        year, month, day = int(row['year']), int(row['month']), int(row['day'])
        
        # 构建日期字符串
        date_str = f"{year:04d}{month:02d}{day:02d}"
        
        # 为每个变量匹配对应的tif文件
        for col_name, var_name in VARIABLES.items():
            tif_path = os.path.join(MLD_BASE_PATH, var_name, f"{var_name}MLDaver{date_str}.tif")
            
            if os.path.exists(tif_path):
                pixel_value = get_pixel_value_from_tif(tif_path, lat, lon)
                df.at[idx, col_name] = pixel_value
            else:
                print(f"File not found: {tif_path}")
    
    # 输出文件
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = f"/home/yytan/Downloads/{base_name}_mldaver.csv"
    df.to_csv(output_path, index=False)
    print(f"Output saved to: {output_path}")
    
    return df

def main():
    """主函数"""
    for input_file in INPUT_FILES:
        if os.path.exists(input_file):
            match_mld_average_for_file(input_file)
        else:
            print(f"Input file not found: {input_file}")

if __name__ == '__main__':
    main() 
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

    # TOA计算
    toa_matrix = np.full(sic.shape[1:], np.nan)
    if min_date_this is not None and min_date_next is not None:
        start_idx = (min_date_this - base_date).days
        end_idx = (min_date_next - base_date).days
        for lat in range(sic.shape[1]):
            for lon in range(sic.shape[2]):
                for i in range(start_idx, min(end_idx, days_in_year)):
                    if sic[i, lat, lon] > 0.15:
                        toa_matrix[lat, lon] = get_julian_day(base_date + datetime.timedelta(days=i))
                        break

    # TOR计算
    tor_matrix = np.full(sic.shape[1:], np.nan)
    if min_date_this is not None and min_date_prev is not None:
        start_idx = (min_date_this - base_date).days
        end_idx = (min_date_prev - base_date).days
        for lat in range(sic.shape[1]):
            for lon in range(sic.shape[2]):
                for i in range(start_idx, max(end_idx, 0), -1):
                    if sic[i, lat, lon] > 0.15:
                        # 如果日期在1992年，则计算相对于1993年1月1日的负天数
                        target_date = base_date + datetime.timedelta(days=i)
                        if target_date.year < year:
                            # 计算从1993年1月1日往前推的天数（负数）
                            days_diff = (target_date - base_date).days
                            tor_matrix[lat, lon] = days_diff
                        else:
                            tor_matrix[lat, lon] = get_julian_day(target_date)
                        break

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
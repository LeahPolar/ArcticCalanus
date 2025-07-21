import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import rasterio
from rasterio.transform import from_origin
from datetime import datetime, timedelta
from tqdm import tqdm  # 新增

# 读取深度表
DEPTH_CSV = '/home/yytan/Downloads/depth.csv'
depth_df = pd.read_csv(DEPTH_CSV)
depth_arr = depth_df.values.flatten()  # 假设只有一列

# 输出文件夹
OUTPUT_BASE = '/home/yytan/Variables/'
VARIABLES = {'CHL': 'phyto', 'SST': 'temp', 'Sal': 'salinity'}

# GLORYS2v4数据路径模板
glorys_path_tpl = '/data/oceandata/GLORYS2v4Combine/GLORYS2v4_{year}.nc'
# MLD数据路径模板
mld_path_tpl = '/data/oceandata/GLOBAL_MULTIYEAR_PHY_ENS_001_031/cmems_mod_glo_phy-all_my_0.25deg_P1D-m/{year:04d}/{month:02d}/cmems_mod_glo_phy-all_my_0.25deg_P1D-m-{year:04d}{month:02d}{day:02d}.nc'

# 创建输出文件夹
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for var in VARIABLES:
    ensure_dir(os.path.join(OUTPUT_BASE, var))

def find_nearest_depth_idx(depths, mld):
    """找到最接近MLD的深度索引"""
    return np.abs(depths - mld).argmin()

def trapezoidal_mean(profile, depths, mld):
    """对profile[0:idx+1]和depths[0:idx+1]用梯形法则积分，idx为最接近mld的索引"""
    idx = find_nearest_depth_idx(depths, mld)
    if idx == 0:
        return profile[0]
    # 只积分到MLD深度
    depths_ext = np.append(depths[:idx], mld)
    profile_ext = np.append(profile[:idx], np.interp(mld, depths[:idx+1], profile[:idx+1]))
    integral = np.trapz(profile_ext, depths_ext)
    return integral / mld if mld > 0 else np.nan

def process_one_day(glorys_ds, day_idx, mld_file, depth_arr, varname):
    # 读取变量的全层数据 (z, y, x)
    data = glorys_ds.variables[varname][day_idx, :, :, :]  # (z, y, x)
    # 读取MLD (y, x)
    mld = mld_file.variables['mlotst_glor'][0, -201:, :]  # (y, x)
    out = np.full_like(mld, np.nan, dtype=np.float32)
    for i in range(mld.shape[0]):
        for j in range(mld.shape[1]):
            mld_val = mld[i, j]
            if np.isnan(mld_val) or mld_val <= 0:
                continue
            profile = data[:, i, j]
            if np.any(np.isnan(profile)):
                continue
            out[i, j] = trapezoidal_mean(profile, depth_arr, mld_val)
    return out

def main():
    for year in tqdm(range(1993, 2023), desc='年份进度'):
        print(f"Processing year {year}")
        nc_file = glorys_path_tpl.format(year=year)
        if not os.path.exists(nc_file):
            print(f"File not found: {nc_file}")
            continue
        ds = nc.Dataset(nc_file)
        # 获取该年有多少天
        time_var = ds.variables['time']
        n_days = len(time_var)
        base_date = datetime(year, 1, 1)
        for i in tqdm(range(n_days), desc=f'{year}年天数进度', leave=False):
            target_date = base_date + timedelta(days=i)
            print(f"  {target_date:%Y-%m-%d}")
            mld_path = mld_path_tpl.format(year=target_date.year, month=target_date.month, day=target_date.day)
            if not os.path.exists(mld_path):
                print(f"    MLD file not found: {mld_path}")
                continue
            mld_file = nc.Dataset(mld_path)
            for var, ncvar in VARIABLES.items():
                out_arr = process_one_day(ds, i, mld_file, depth_arr, ncvar)
                out_arr = np.flipud(out_arr)
                # 输出tif
                output_dir = os.path.join(OUTPUT_BASE, var)
                output_path = os.path.join(output_dir, f"{var}MLDaver{target_date:%Y%m%d}.tif")
                transform = from_origin(-180, 40, 0.25, 0.25)
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=out_arr.shape[0],
                    width=out_arr.shape[1],
                    count=1,
                    dtype=rasterio.float32,
                    crs='EPSG:4326',
                    transform=transform
                ) as dst:
                    dst.write(out_arr, 1)
            mld_file.close()
        ds.close()

if __name__ == '__main__':
    main() 
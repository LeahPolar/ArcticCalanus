import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import rasterio
from rasterio.transform import from_origin
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numba

DEPTH_CSV = '/home/yytan/Downloads/depth.csv'
depth_df = pd.read_csv(DEPTH_CSV)
depth_arr = depth_df.values.flatten()
# 确保depth_arr是普通numpy数组
if hasattr(depth_arr, 'mask'):
    depth_arr = depth_arr.filled(np.nan)
depth_arr = np.asarray(depth_arr, dtype=np.float64)

OUTPUT_BASE = '/home/yytan/Variables/'
VARIABLES = {'CHL': 'phyto', 'SST': 'temp', 'Sal': 'salinity'}

glorys_path_tpl = '/data/oceandata/GLORYS2v4Combine/GLORYS2v4_{year}.nc'
mld_path_tpl = '/data/oceandata/GLOBAL_MULTIYEAR_PHY_ENS_001_031/cmems_mod_glo_phy-all_my_0.25deg_P1D-m/{year:04d}/{month:02d}/cmems_mod_glo_phy-all_my_0.25deg_P1D-m-{year:04d}{month:02d}{day:02d}.nc'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for var in VARIABLES:
    ensure_dir(os.path.join(OUTPUT_BASE, var))

@numba.njit
def find_nearest_depth_idx_numba(depths, mld):
    return np.abs(depths - mld).argmin()

@numba.njit
def trapezoidal_mean_numba(profile, depths, mld):
    idx = find_nearest_depth_idx_numba(depths, mld)
    if idx == 0:
        return profile[0]
    depths_ext = np.append(depths[:idx], mld)
    profile_ext = np.append(profile[:idx], np.interp(mld, depths[:idx+1], profile[:idx+1]))
    integral = np.trapz(profile_ext, depths_ext)
    return integral / mld if mld > 0 else np.nan

def process_one_day_mp(args):
    year, day_idx, depth_arr = args
    nc_file = glorys_path_tpl.format(year=year)
    ds = nc.Dataset(nc_file)
    time_var = ds.variables['time']
    base_date = datetime(year, 1, 1)
    target_date = base_date + timedelta(days=day_idx)
    mld_path = mld_path_tpl.format(year=target_date.year, month=target_date.month, day=target_date.day)
    if not os.path.exists(mld_path):
        ds.close()
        return None
    mld_file = nc.Dataset(mld_path)
    for var, ncvar in VARIABLES.items():
        data = ds.variables[ncvar][day_idx, :, :, :]
        mld = mld_file.variables['mlotst_glor'][0, -201:, :]
        out = np.full_like(mld, np.nan, dtype=np.float32)
        for i in range(mld.shape[0]):
            for j in range(mld.shape[1]):
                mld_val = mld[i, j]
                if np.isnan(mld_val) or mld_val <= 0:
                    continue
                profile = data[:, i, j]
                if np.any(np.isnan(profile)):
                    continue
                # 将MaskedArray转换为普通numpy数组
                if hasattr(profile, 'mask'):
                    profile_np = profile.filled(np.nan)
                else:
                    profile_np = profile
                profile_np = np.asarray(profile_np, dtype=np.float64)
                mld_val_np = float(mld_val) if not np.isnan(mld_val) else np.nan
                out[i, j] = trapezoidal_mean_numba(profile_np, depth_arr, mld_val_np)
        out = np.flipud(out)
        output_dir = os.path.join(OUTPUT_BASE, var)
        output_path = os.path.join(output_dir, f"{var}MLDaver{target_date:%Y%m%d}.tif")
        transform = from_origin(-180, 40, 0.25, 0.25)
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=out.shape[0],
            width=out.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(out, 1)
    mld_file.close()
    ds.close()
    return True

def main():
    for year in tqdm(range(1997, 2023), desc='年份进度'):
        nc_file = glorys_path_tpl.format(year=year)
        if not os.path.exists(nc_file):
            continue
        ds = nc.Dataset(nc_file)
        n_days = len(ds.variables['time'])
        ds.close()
        args_list = [(year, i, depth_arr) for i in range(n_days)]
        with ProcessPoolExecutor(max_workers=4) as executor:
            list(tqdm(executor.map(process_one_day_mp, args_list), total=n_days, desc=f'{year}年天数进度', leave=False))

if __name__ == '__main__':
    main() 
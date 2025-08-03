# -*- coding: utf-8 -*-
"""
Created on Mon May 26 20:21:24 2025

@author: Leah
"""

import pandas as pd
import xarray as xr
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv("/home/yytan/Downloads/final_standardizedMOSAiC.csv")
# df = pd.read_csv("D:/0-Calanus/0-SurveyData/0-OBIS/Calanus glacialis/CG-Occurrence-55N.csv")
columns_to_update = ["SIC", "CHL", "SST", "Sal", 'u', "v", "npp", "nitrat", "SIT", "mlotst_glor"]
df[columns_to_update]=np.nan


for i, row in tqdm(df.iterrows()):
    lat = row['Latitude']
    lon = row['Longitude']
    year = int(row['year'])
    month = int(row['month'])
    day = int(row['day'])
    filepath = f"/data/oceandata/GLOBAL_MULTIYEAR_PHY_ENS_001_031/cmems_mod_glo_phy-all_my_0.25deg_P1D-m/{year:04d}/{month:02d}/"
    filename = f"cmems_mod_glo_phy-all_my_0.25deg_P1D-m-{year:04d}{month:02d}{day:02d}.nc"
    filepath2 = f"/data/oceandata/GLOBAL_MULTIYEAR_BGC_001_029/{year:04d}/{month:02d}/"
    filename2 = f"mercatorfreebiorys2v4_global_mean_{year:04d}{month:02d}{day:02d}.nc"
    try:
        ds = xr.open_dataset(filepath+filename)
        
        selected_mld = ds['mlotst_glor'].sel(
           latitude=lat,
           longitude=lon,
           method="nearest"
        )
        df.at[i, 'mlotst_glor'] = selected_mld.values.item()
        
        selected_sic = ds['siconc_glor'].sel(
           latitude=lat,
           longitude=lon,
           method="nearest"
        )
        df.at[i, 'SIC'] = selected_sic.values.item()
        
        selected_sit = ds['sithick_glor'].sel(
           latitude=lat,
           longitude=lon,
           method="nearest"
        )
        df.at[i, 'SIT'] = selected_sit.values.item()
        
        selected_sal = ds['so_glor'].sel(
           latitude=lat,
           longitude=lon,
           depth=ds.depth[0],
           method="nearest"
        )
        df.at[i, 'Sal'] = selected_sal.values.item()
        
        selected_sst = ds['thetao_glor'].sel(
           latitude=lat,
           longitude=lon,
           depth=ds.depth[0],
           method="nearest"
        )
        df.at[i, 'SST'] = selected_sst.values.item()
        
        selected_u = ds['uo_glor'].sel(
           latitude=lat,
           longitude=lon,
           depth=ds.depth[0],
           method="nearest"
        )
        df.at[i, 'u'] = selected_u.values.item()
        
        selected_v = ds['vo_glor'].sel(
           latitude=lat,
           longitude=lon,
           depth=ds.depth[0],
           method="nearest"
        )
        df.at[i, 'v'] = selected_v.values.item()
        
        ds.close()
        
        ds_env = xr.open_dataset(filepath2+filename2)
        selected_chl = ds_env['chl'].sel(
           latitude=lat,
           longitude=lon,
           depth=ds_env.depth[0],  # 最表层
           method="nearest"
        )
        selected_no3 = ds_env['no3'].sel(
           latitude=lat,
           longitude=lon,
           depth=ds_env.depth[0],  # 最表层
           method="nearest"
        )
        selected_nppv = ds_env['nppv'].sel(
           latitude=lat,
           longitude=lon,
           depth=ds_env.depth[0],  # 最表层
           method="nearest"
        )
       
       # 提取 o2、po4 和 si 值
        df.at[i, 'CHL'] = selected_chl.values.item()
        df.at[i, 'nitrat'] = selected_no3.values.item()
        df.at[i, 'npp'] = selected_nppv.values.item()
       
       # 关闭数据集
        ds_env.close()
        
    except FileNotFoundError:
        print(f"File not found for row {i} with date {year}-{month:02d}-{day:02d}")
        df.loc[i, columns_to_update] = np.nan
    except Exception as e:
        print(f"Error retrieving data for row {i} (lat: {lat}, lon: {lon}, date: {year}-{month:02d}-{day:02d}): {e}")
        df.loc[i, columns_to_update] = np.nan

df.to_csv("/home/yytan/Downloads/FinalMatchuped6-MOSAiC.csv", index=False)
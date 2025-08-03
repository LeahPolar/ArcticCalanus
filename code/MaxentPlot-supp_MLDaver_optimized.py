# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:18:24 2024

@author: Yiyang Tan
"""

import pandas as pd
import numpy as np
import elapid
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin
import warnings
from matplotlib import gridspec
import glob
import os
import matplotlib
import matplotlib.ticker as mticker
from scipy.stats import linregress
import cartopy.mpl.ticker as ctk

matplotlib.use('Agg')  # 使用Agg后端

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def load_and_prepare_data(file_path):
    """加载和准备数据"""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # 数据验证
    required_columns = ['Cgla_c1c2', 'Sal', 'MLDaverSST', 'SST', 'SIC', 'mlotst_glor', 'MLDaverSal']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 创建目标变量
    df["Cgla_occur"] = (df["Cgla_c1c2"] != 0).astype(int)
    df["cycle_year"] = df['year']
    
    # 移除包含NaN的行
    feature_names = ['Sal', 'MLDaverSST', 'SST', 'SIC', 'mlotst_glor', 'MLDaverSal']
    df_clean = df.dropna(subset=feature_names + ['Cgla_occur'])
    
    print(f"Original data shape: {df.shape}")
    print(f"Clean data shape: {df_clean.shape}")
    
    return df_clean, feature_names

def train_maxent_model(df, feature_names):
    """训练MaxEnt模型"""
    print("Training MaxEnt model...")
    X = df[feature_names]
    y = df["Cgla_occur"]
    maxent = elapid.MaxentModel()
    maxent.fit(X, y)
    return maxent

def load_mld_average_data(target_date, base_shape=(201, 1440)):
    """加载MLD平均数据"""
    mld_sst_path = f"/home/yytan/Variables/SST/SSTMLDaver{target_date:%Y%m%d}.tif"
    mld_sal_path = f"/home/yytan/Variables/Sal/SalMLDaver{target_date:%Y%m%d}.tif"
    
    MLDaverSST = np.full(base_shape, np.nan)
    MLDaverSal = np.full(base_shape, np.nan)
    
    try:
        if os.path.exists(mld_sst_path):
            with rasterio.open(mld_sst_path) as src:
                MLDaverSST = src.read(1)
                MLDaverSST = np.flipud(MLDaverSST)
    except Exception as e:
        print(f"Warning: Error loading MLDaverSST for {target_date:%Y%m%d}: {e}")
    
    try:
        if os.path.exists(mld_sal_path):
            with rasterio.open(mld_sal_path) as src:
                MLDaverSal = src.read(1)
                MLDaverSal = np.flipud(MLDaverSal)
    except Exception as e:
        print(f"Warning: Error loading MLDaverSal for {target_date:%Y%m%d}: {e}")
    
    return MLDaverSST, MLDaverSal

def MaxentPredict(outputyear, maxent, feature_names):
    """优化的MaxEnt预测函数"""
    nc_file = f'/data/oceandata/GLORYS2v4Combine/GLORYS2v4_{outputyear}.nc'
    
    if not os.path.exists(nc_file):
        print(f"Warning: GLORYS2v4 file not found for year {outputyear}")
        return
    
    try:
        ds = nc.Dataset(nc_file)
    except Exception as e:
        print(f"Error opening GLORYS2v4 file for year {outputyear}: {e}")
        return
    
    base_date = datetime.datetime(outputyear, 1, 1)
    output_path = f"/home/yytan/Maxent/Cgla-supp-MLDaver/{outputyear}"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    days_in_year = (datetime.datetime(outputyear, 12, 31) - datetime.datetime(outputyear, 1, 1)).days + 1
    
    # 预分配数组以提高效率
    base_shape = (201, 1440)
    
    for i in tqdm(range(days_in_year), desc=f"Processing year {outputyear}"):
        target_date = base_date + datetime.timedelta(days=i)
        
        try:
            # 读取基础变量
            SIC = ds.variables['sic'][i,:,:]
            SST = ds.variables['temp'][i,0,:,:]
            Sal = ds.variables['salinity'][i,0,:,:]
            nitrat = ds.variables['nitrat'][i,0,:,:]
            
            # 读取MLD文件
            filepath = f"/data/oceandata/GLOBAL_MULTIYEAR_PHY_ENS_001_031/cmems_mod_glo_phy-all_my_0.25deg_P1D-m/{target_date.year:04d}/{target_date.month:02d}/"
            filename = f"cmems_mod_glo_phy-all_my_0.25deg_P1D-m-{target_date.year:04d}{target_date.month:02d}{target_date.day:02d}.nc"
            
            if not os.path.exists(filepath+filename):
                continue
                
            try:
                phy = nc.Dataset(filepath+filename)
                mld = phy.variables["mlotst_glor"][0, -201:,:]
                phy.close()
            except Exception as e:
                print(f"Warning: Error reading MLD file for {target_date:%Y%m%d}: {e}")
                continue
            
            # 加载MLD平均数据
            MLDaverSST, MLDaverSal = load_mld_average_data(target_date, base_shape)
            
            # 创建环境变量DataFrame - 优化版本
            Envivar = pd.DataFrame({
                "SIC": SIC.reshape(-1),
                'SST': SST.reshape(-1),
                'Sal': Sal.reshape(-1),
                'nitrat': nitrat.reshape(-1),
                'mlotst_glor': mld.reshape(-1),
                'MLDaverSST': MLDaverSST.reshape(-1),
                'MLDaverSal': MLDaverSal.reshape(-1)
            })
            
            # 预测
            Cgla_suit = maxent.predict(Envivar)
            Cgla_suit = Cgla_suit.reshape(base_shape)
            Cgla_suit = np.flipud(Cgla_suit)
            
            # 保存tif文件
            transform = from_origin(-180, 40, 0.25, 0.25)
            with rasterio.open(
                    output_path+f"/CglaSuitability{target_date:%Y%m%d}.tif",
                    'w',
                    driver = 'GTiff',
                    height=Cgla_suit.shape[0],
                    width=Cgla_suit.shape[1],
                    count=1,
                    dtype=rasterio.float32,
                    crs='EPSG:4326',
                    transform=transform
                    ) as dst:
                dst.write(Cgla_suit, 1)
            
            # 绘制图像 - 保持原有代码不变
            fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(8, 8))
            ax.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())
            img = ax.imshow(Cgla_suit[:-60,:], transform=ccrs.PlateCarree(), extent=(-180, 180, 55, 90),
                            origin='upper', cmap='jet', vmin=0, vmax=1, zorder=1)
            ax.set_boundary(circle, transform=ax.transAxes)
            ax.add_feature(cfeature.LAND, facecolor='gray',zorder=2)
            ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder=3)
            gl = ax.gridlines(draw_labels=True)
            gl.xlabel_style = {'size': 12, 'color': 'black'}
            gl.ylabel_style = {'size': 12, 'color': 'black'}
            gl.ylocator = mticker.FixedLocator(range(60, 91, 10))
            cbar = plt.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label("Calanus glacialis habitat suitability "+target_date.__format__("%Y%m%d"))
            plt.savefig(output_path+f"/CglaSuitability{target_date:%Y%m%d}.jpg", dpi=300)
            plt.close('all')
            
        except Exception as e:
            print(f"Error processing day {target_date:%Y%m%d}: {e}")
            continue
    
    ds.close()

def load_tif_files(file_pattern):
    """Load multiple TIFF files matching the file pattern and return a dictionary of seasonal arrays."""
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"Warning: No files found matching pattern {file_pattern}")
        return {}, None
        
    seasons = {'JFM':[], 'AMJ': [], 'JAS': [], 'OND':[]}
    for file in tqdm(files, desc="Loading TIF files"):
        try:
            with rasterio.open(file) as src:
                data = src.read(1)
                data = np.where((data >= 0) & (data <= 1), data, np.nan)  # 过滤数据
                date_str = os.path.basename(file)[15:23]
                date = datetime.datetime.strptime(date_str, "%Y%m%d")
                if  1 <= date.month <= 3:
                    season = 'JFM'
                elif 4 <= date.month <= 6:
                    season = 'AMJ'
                elif 7 <= date.month <= 9:
                    season = 'JAS'
                elif 10 <= date.month <= 12:
                    season = "OND"
                seasons[season].append(data)
        except Exception as e:
            print(f"Warning: Error loading file {file}: {e}")
            continue

    for season in seasons:
        if seasons[season]:  # 确保有数据
            seasons[season] = np.array(seasons[season])

    return seasons, files[0] if files else None

def calculate_seasonal_average(seasonal_data):
    """Calculate the seasonal average considering NaN values."""
    seasonal_averages = {}
    for season, data in seasonal_data.items():
        if len(data) > 0:  # 确保有数据
            seasonal_average = np.nanmean(data, axis=0)
            seasonal_averages[season] = seasonal_average
    return seasonal_averages

def save_tif(file_path, template_file, data):
    """Save the average data to a new TIFF file using the template_file for metadata."""
    try:
        with rasterio.open(template_file) as src:
            profile = src.profile
            profile.update(dtype=rasterio.float64, count=1, compress='lzw')

        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(data.astype(rasterio.float64), 1)
    except Exception as e:
        print(f"Error saving TIF file {file_path}: {e}")

def SeasonalOutput(year, outputpath):
    file_pattern = f"/home/yytan/Maxent/Cgla-supp-MLDaver/{year}/CglaSuitability*.tif"
    seasonal_data, template_file = load_tif_files(file_pattern)
    
    if not template_file:
        print(f"Warning: No template file found for year {year}")
        return
        
    seasonal_averages = calculate_seasonal_average(seasonal_data)
    
    for season, average_data in seasonal_averages.items():
        output_file = outputpath+ f"/{season}_{year}_suitability.tif"
        save_tif(output_file, template_file, average_data)
    
    # 绘制季节图 - 保持原有代码不变
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2,  width_ratios=[1, 1], height_ratios=[1, 1])
    gs.update(hspace=0.3)
    extent = [-180, 180, 55, 90]
    i = 0
    axe = []
    for season, average in seasonal_averages.items():
        ax = fig.add_subplot(gs[i], projection=projection)
        ax.set_extent(extent, crs=ccrs.PlateCarree())  # 设置经纬度范围
        img = ax.imshow(average[:-60, :], transform=ccrs.PlateCarree(), extent=extent,
                        origin='upper', cmap='jet', vmin=0,vmax=1, zorder=1)
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.add_feature(cfeature.LAND, facecolor='gray', zorder=2)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder=3)
        ax.set_title(f'{season}', loc='left')
        gl = ax.gridlines(draw_labels=True)
        gl.xlabel_style = {'size': 12, 'color': 'black'}
        gl.ylabel_style = {'size': 12, 'color': 'black'}
        gl.ylocator = mticker.FixedLocator(range(60, 91, 10))
        i+=1
        axe.append(ax)
    cbar = fig.colorbar(img, ax=axe, orientation='horizontal', fraction=0.046, pad=0.1)
    cbar.set_label(f'Cgla {year} Suitability', fontsize=12)
    plt.savefig(f"/home/yytan/Figures/Cgla-supp-MLDaver/Cgla{year}Siutability.jpg", dpi=300)
    plt.close("all")

def calculate_long_term_trends():
    """计算长期趋势 - 优化版本"""
    print("Calculating long-term trends...")
    seasons = {'JFM': [], 'AMJ': [], 'JAS': [], 'OND': []}
    
    for season in seasons:
        years = list(range(1993, 2023))
        input_dir = "/home/yytan/Maxent/Cgla-supp-MLDaver/seasonal"
        tif_files = [os.path.join(input_dir, f"{season}_{year}_suitability.tif") for year in years]
        
        # 检查文件是否存在
        existing_files = [f for f in tif_files if os.path.exists(f)]
        if not existing_files:
            print(f"Warning: No files found for season {season}")
            continue
            
        try:
            with rasterio.open(existing_files[0]) as src:
                rows, cols = src.shape
                all_data = np.zeros((len(existing_files), rows, cols), dtype=np.float32)
            
            # 并行读取文件
            for i, tif_file in enumerate(existing_files):
                try:
                    with rasterio.open(tif_file) as src:
                        all_data[i, :, :] = src.read(1)
                except Exception as e:
                    print(f"Warning: Error reading {tif_file}: {e}")
                    all_data[i, :, :] = np.nan
            
            mean_data = np.nanmean(all_data, axis=0)
            trend_array = np.full((rows, cols), np.nan, dtype=np.float64)
            years_array = np.array([int(os.path.basename(f).split('_')[1]) for f in existing_files])
            
            # 向量化计算趋势
            for i in range(rows):
                for j in range(cols):
                    pixel_series = all_data[:, i, j]
                    valid_mask = ~np.isnan(pixel_series)
                    if np.sum(valid_mask) >= 2:  # 如果有效点至少2个
                        try:
                            slope, intercept, r_value, p_value, std_err = linregress(years_array[valid_mask], pixel_series[valid_mask])
                            if p_value < 0.05:  # 只保留 p-value < 0.05 的像素值
                                trend_array[i, j] = slope
                        except Exception as e:
                            continue
            
            seasons[season].append(mean_data)
            seasons[season].append(trend_array)
            
        except Exception as e:
            print(f"Error processing season {season}: {e}")
            continue
    
    return seasons

def main():
    """主函数"""
    # 加载数据
    data_path = "/home/yytan/Downloads/Cgla_c1c2_mldaver.csv"
    df, feature_names = load_and_prepare_data(data_path)
    
    # 训练模型
    maxent = train_maxent_model(df, feature_names)
    
    # 设置投影参数
    global theta, center, radius, verts, circle, projection
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    projection = ccrs.NorthPolarStereo()
    
    # 创建输出目录
    outputpath = "/home/yytan/Maxent/Cgla-supp-MLDaver/seasonal"
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    os.makedirs("/home/yytan/Figures/Cgla-supp-MLDaver", exist_ok=True)
    
    # 主循环
    for i in range(1993, 2023):
        try:
            MaxentPredict(i, maxent, feature_names)
            SeasonalOutput(i, outputpath)
            print(f"year {i} is finished")
        except Exception as e:
            print(f"Error processing year {i}: {e}")
            continue
    
    # 计算长期趋势
    seasons = calculate_long_term_trends()
    
    # 绘制长期趋势图 - 保持原有代码不变
    title = ["mean", "trend"]

    # 大西洋区域图
    leftlon, rightlon, lowerlat, upperlat = -30, 60, 60, 85
    rect = mpath.Path([[leftlon, lowerlat], [rightlon, lowerlat],
        [rightlon, upperlat], [leftlon, upperlat], [leftlon, lowerlat]]).interpolated(50)

    proj=ccrs.NearsidePerspective(central_longitude=(leftlon+rightlon)*0.5,
        central_latitude=(lowerlat+upperlat)*0.5)

    for j, variable in enumerate(title):
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        gs.update(hspace=0.3)
        extent = [-180, 180, 55, 90]
        
        i = 0
        axe = []
        for season, diff in seasons.items():
            if len(diff) < 2:  # 确保有数据
                continue
            ax = fig.add_subplot(gs[i], projection=proj)
            
            if j == 0:
                img = ax.imshow(seasons[season][j][:-60, :], transform=ccrs.PlateCarree(), extent=extent,
                              origin='upper', cmap='jet', vmin=0, vmax=1, zorder=1)
            elif j == 1:
                img = ax.imshow(seasons[season][j][:-60, :], transform=ccrs.PlateCarree(),
                                extent=extent,
                                origin='upper', cmap='coolwarm', vmin=-0.03, vmax=0.03, zorder=1)
            ax.add_feature(cfeature.COASTLINE.with_scale('110m'))
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
            rect_in_target = proj_to_data.transform_path(rect)
            ax.set_boundary(rect_in_target)
            ax.set_xlim(rect_in_target.vertices[:,0].min(), rect_in_target.vertices[:,0].max())
            ax.set_ylim(rect_in_target.vertices[:,1].min(), rect_in_target.vertices[:,1].max())
            ax.set_title(f'{season}', loc='left')
            gl=ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linestyle='dashed')
            gl.top_labels=False
            gl.right_labels=False
            gl.rotate_labels=False
            gl.xformatter=ctk.LongitudeFormatter(zero_direction_label=True)
            gl.yformatter=ctk.LatitudeFormatter()
            i+=1
            axe.append(ax)
        if j != 1: ext ="neither" 
        else: ext = "both"
        cbar = fig.colorbar(img, ax=axe, orientation='horizontal', fraction=0.046, pad=0.1, extend=ext)
        cbar.set_label(f'Cgla 1993-2022 Suitability {variable}', fontsize=12)
        plt.savefig(f"/home/yytan/Figures/Cgla-supp-MLDaver/CglaSeasonal{variable}Atlantic_MLDaver.jpg", dpi=300)

    # 太平洋区域图
    leftlon, rightlon, lowerlat, upperlat = -180, -120, 55, 80
    rect = mpath.Path([[leftlon, lowerlat], [rightlon, lowerlat],
        [rightlon, upperlat], [leftlon, upperlat], [leftlon, lowerlat]]).interpolated(50)

    proj=ccrs.NearsidePerspective(central_longitude=(leftlon+rightlon)*0.5,
        central_latitude=(lowerlat+upperlat)*0.5)

    for j, variable in enumerate(title):
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        gs.update(hspace=0.3)
        extent = [-180, 180, 55, 90]
        i = 0
        axe = []
        for season, diff in seasons.items():
            if len(diff) < 2:  # 确保有数据
                continue
            ax = fig.add_subplot(gs[i], projection=proj)
            
            if j == 0:
                img = ax.imshow(seasons[season][j][:-60, :], transform=ccrs.PlateCarree(), extent=extent,
                              origin='upper', cmap='jet', vmin=0, vmax=1, zorder=1)
            elif j == 1:
                img = ax.imshow(seasons[season][j][:-60, :], transform=ccrs.PlateCarree(),
                                extent=extent,
                                origin='upper', cmap='coolwarm', vmin=-0.03, vmax=0.03, zorder=1)
            ax.add_feature(cfeature.COASTLINE.with_scale('110m'), zorder=3)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
            proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
            rect_in_target = proj_to_data.transform_path(rect)
            ax.set_boundary(rect_in_target)
            ax.set_xlim(rect_in_target.vertices[:,0].min(), rect_in_target.vertices[:,0].max())
            ax.set_ylim(rect_in_target.vertices[:,1].min(), rect_in_target.vertices[:,1].max())
            ax.set_title(f'{season}', loc='left')
            gl=ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linestyle='dashed')
            gl.top_labels=False
            gl.right_labels=False
            gl.rotate_labels=False
            gl.xformatter=ctk.LongitudeFormatter(zero_direction_label=True)
            gl.yformatter=ctk.LatitudeFormatter()
            i+=1
            axe.append(ax)
        if j != 1: ext ="neither" 
        else: ext = "both"
        cbar = fig.colorbar(img, ax=axe, orientation='horizontal', fraction=0.046, pad=0.1, extend=ext)
        cbar.set_label(f'Cgla 1993-2022 Suitability {variable}', fontsize=12)
        plt.savefig(f"/home/yytan/Figures/Cgla-supp-MLDaver/CglaSeasonal{variable}Pacific_MLDaver.jpg", dpi=300)

if __name__ == '__main__':
    main() 
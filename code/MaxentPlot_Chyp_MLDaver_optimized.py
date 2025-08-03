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
import re

matplotlib.use('Agg')  # 使用Agg后端

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ============================================================================
# 第一部分：MaxEnt预测和季节分析
# ============================================================================

def load_and_prepare_data(file_path):
    """加载和准备数据"""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # 数据验证 - 使用Chyp的特征变量
    required_columns = ['Chyp_c1c2', 'MLDaverSST', 'CHL', 'MLDaverCHL', 'SST', 'MLDaverSal', 'nitrat']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 创建目标变量
    df["Chyp_occur"] = (df["Chyp_c1c2"] != 0).astype(int)
    df["cycle_year"] = df['year']
    
    # 移除包含NaN的行 - 使用Chyp的特征变量
    feature_names = ['MLDaverSST', 'CHL', 'MLDaverCHL', 'SST', 'MLDaverSal', 'nitrat']
    df_clean = df.dropna(subset=feature_names + ['Chyp_occur'])
    
    print(f"Original data shape: {df.shape}")
    print(f"Clean data shape: {df_clean.shape}")
    
    return df_clean, feature_names

def train_maxent_model(df, feature_names):
    """训练MaxEnt模型"""
    print("Training MaxEnt model...")
    X = df[feature_names]
    y = df["Chyp_occur"]
    maxent = elapid.MaxentModel()
    maxent.fit(X, y)
    return maxent

def load_mld_average_data(target_date, base_shape=(201, 1440)):
    """加载MLD平均数据"""
    mld_chl_path = f"/home/yytan/Variables/CHL/CHLMLDaver{target_date:%Y%m%d}.tif"
    mld_sst_path = f"/home/yytan/Variables/SST/SSTMLDaver{target_date:%Y%m%d}.tif"
    mld_sal_path = f"/home/yytan/Variables/Sal/SalMLDaver{target_date:%Y%m%d}.tif"
    
    MLDaverCHL = np.full(base_shape, np.nan)
    MLDaverSST = np.full(base_shape, np.nan)
    MLDaverSal = np.full(base_shape, np.nan)
    
    try:
        if os.path.exists(mld_chl_path):
            with rasterio.open(mld_chl_path) as src:
                MLDaverCHL = src.read(1)
                MLDaverCHL = np.flipud(MLDaverCHL)
    except Exception as e:
        print(f"Warning: Error loading MLDaverCHL for {target_date:%Y%m%d}: {e}")
    
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
    
    return MLDaverCHL, MLDaverSST, MLDaverSal

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
    output_path = f"/home/yytan/Maxent/Chyp-supp-MLDaver/{outputyear}"
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
            CHL = ds.variables['phyto'][i,0,:,:]
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
            MLDaverCHL, MLDaverSST, MLDaverSal = load_mld_average_data(target_date, base_shape)
            
            # 创建环境变量DataFrame - 使用Chyp的特征变量
            Envivar = pd.DataFrame({
                "SIC": SIC.reshape(-1),
                'SST': SST.reshape(-1),
                'CHL': CHL.reshape(-1),
                'nitrat': nitrat.reshape(-1),
                'mlotst_glor': mld.reshape(-1),
                'MLDaverSST': MLDaverSST.reshape(-1),
                'MLDaverCHL': MLDaverCHL.reshape(-1),
                'MLDaverSal': MLDaverSal.reshape(-1)
            })
            
            # 预测
            Chyp_suit = maxent.predict(Envivar)
            Chyp_suit = Chyp_suit.reshape(base_shape)
            Chyp_suit = np.flipud(Chyp_suit)
            
            # 保存tif文件
            transform = from_origin(-180, 40, 0.25, 0.25)
            with rasterio.open(
                    output_path+f"/ChypSuitability{target_date:%Y%m%d}.tif",
                    'w',
                    driver = 'GTiff',
                    height=Chyp_suit.shape[0],
                    width=Chyp_suit.shape[1],
                    count=1,
                    dtype=rasterio.float32,
                    crs='EPSG:4326',
                    transform=transform
                    ) as dst:
                dst.write(Chyp_suit, 1)
            
            # 绘制图像 - 保持原有代码不变
            fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(8, 8))
            ax.set_extent([-180, 180, 55, 90], crs=ccrs.PlateCarree())
            img = ax.imshow(Chyp_suit[:-60,:], transform=ccrs.PlateCarree(), extent=(-180, 180, 55, 90),
                            origin='upper', cmap='jet', vmin=0, vmax=1, zorder=1)
            ax.set_boundary(circle, transform=ax.transAxes)
            ax.add_feature(cfeature.LAND, facecolor='gray',zorder=2)
            ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder=3)
            gl = ax.gridlines(draw_labels=True)
            gl.xlabel_style = {'size': 12, 'color': 'black'}
            gl.ylabel_style = {'size': 12, 'color': 'black'}
            gl.ylocator = mticker.FixedLocator(range(60, 91, 10))
            cbar = plt.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label("Calanus hyperboreus habitat suitability "+target_date.__format__("%Y%m%d"))
            plt.savefig(output_path+f"/ChypSuitability{target_date:%Y%m%d}.jpg", dpi=300)
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
    file_pattern = f"/home/yytan/Maxent/Chyp-supp-MLDaver/{year}/ChypSuitability*.tif"
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
    cbar.set_label(f'Chyp {year} Suitability', fontsize=12)
    plt.savefig(f"/home/yytan/Figures/Chyp-supp-MLDaver/Chyp{year}Siutability.jpg", dpi=300)
    plt.close("all")

def calculate_long_term_trends():
    """计算长期趋势 - 优化版本"""
    print("Calculating long-term trends...")
    seasons = {'JFM': [], 'AMJ': [], 'JAS': [], 'OND': []}
    
    for season in seasons:
        years = list(range(1993, 2023))
        input_dir = "/home/yytan/Maxent/Chyp-supp-MLDaver/seasonal"
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

def run_maxent_analysis():
    """运行MaxEnt分析"""
    print("=" * 60)
    print("开始MaxEnt预测和季节分析")
    print("=" * 60)
    
    # 加载数据
    data_path = "/home/yytan/Downloads/Chyp_c1c2_mldaver.csv"
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
    outputpath = "/home/yytan/Maxent/Chyp-supp-MLDaver/seasonal"
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    os.makedirs("/home/yytan/Figures/Chyp-supp-MLDaver", exist_ok=True)
    
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
        cbar.set_label(f'Chyp 1993-2022 Suitability {variable}', fontsize=12)
        plt.savefig(f"/home/yytan/Figures/Chyp-supp-MLDaver/ChypSeasonal{variable}Atlantic_MLDaver.jpg", dpi=300)

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
        cbar.set_label(f'Chyp 1993-2022 Suitability {variable}', fontsize=12)
        plt.savefig(f"/home/yytan/Figures/Chyp-supp-MLDaver/ChypSeasonal{variable}Pacific_MLDaver.jpg", dpi=300)

# ============================================================================
# 第二部分：年度均值计算和差异分析
# ============================================================================

def calculate_year_means():
    """计算年度均值"""
    print("=" * 60)
    print("开始年度均值计算和差异分析")
    print("=" * 60)
    
    # 更新输入和输出路径为Chyp版本
    input_path = "/home/yytan/Maxent/Chyp-supp-MLDaver/"
    output_path = "/home/yytan/Maxent/Chyp-supp-MLDaver/"
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    tif_files = []
    for subdir in os.listdir(input_path):
        if re.fullmatch(r"\d{4}", subdir):  # 只匹配4位数字的文件夹
            year_folder = os.path.join(input_path, subdir)
            tif_files.extend(sorted(glob.glob(os.path.join(year_folder, "ChypSuitability*.tif"))))
    
    print(f"Found {len(tif_files)} TIF files")
    
    group1 = []  # 1993-2007
    group2 = []  # 2008-2022
    
    for tif in tif_files:
        # 提取日期
        basename = os.path.basename(tif)
        date_str = basename.replace("ChypSuitability", "").replace(".tif", "")
        year = int(date_str[:4])
        if 1993 <= year <= 2007:
            group1.append(tif)
        elif 2008 <= year <= 2022:
            group2.append(tif)
    
    print(f"Group 1 (1993-2007): {len(group1)} files")
    print(f"Group 2 (2008-2022): {len(group2)} files")
    
    def mean_rasters(file_list):
        if not file_list:
            raise ValueError("输入的文件列表为空，请检查分组条件和文件路径！")
        arr_list = []
        meta = None
        shape0 = None
        for f in file_list:
            with rasterio.open(f) as src:
                arr = src.read(1)
                if shape0 is None:
                    shape0 = arr.shape
                elif arr.shape != shape0:
                    raise ValueError(f"文件 {f} 的shape {arr.shape} 与其他影像不一致，期望shape为 {shape0}！")
                arr = np.where((arr >= 0) & (arr <= 1), arr, np.nan)
                arr_list.append(arr)
                if meta is None:
                    meta = src.meta.copy()
        mean_arr = np.nanmean(arr_list, axis=0)
        return mean_arr, meta
    
    # 计算均值
    print("Calculating mean for group 1 (1993-2007)...")
    mean1, meta1 = mean_rasters(group1)
    
    print("Calculating mean for group 2 (2008-2022)...")
    mean2, meta2 = mean_rasters(group2)
    
    diff = mean2 - mean1
    
    # 计算1993-2022年所有影像的平均
    print("Calculating mean for all years (1993-2022)...")
    all_group = group1 + group2
    mean_all, meta_all = mean_rasters(all_group)
    
    # 保存结果文件 - 更新文件名添加MLDaver后缀
    print("Saving results...")
    with rasterio.open(os.path.join(output_path, "ChypSuitability1993-2007mean_MLDaver.tif"), 'w',
                       driver='GTiff', dtype=rasterio.float32,
                       height=mean1.shape[0], width=mean1.shape[1], count=1,
                       transform=from_origin(-180, 40, 0.25, 0.25),
                       crs='EPSG:4326') as dst:
        dst.write(mean1.astype(np.float32), 1)
    
    with rasterio.open(os.path.join(output_path, "ChypSuitability2008-2022mean_MLDaver.tif"), 'w',
                       driver='GTiff', dtype=rasterio.float32,
                       height=mean2.shape[0], width=mean2.shape[1], count=1,
                       transform=from_origin(-180, 40, 0.25, 0.25),
                       crs='EPSG:4326') as dst:
        dst.write(mean2.astype(np.float32), 1)
    
    with rasterio.open(os.path.join(output_path, "ChypSuitability2008-2022minus1993-2007_MLDaver.tif"), 'w',
                       driver='GTiff', dtype=rasterio.float32,
                       height=diff.shape[0], width=diff.shape[1], count=1,
                       transform=from_origin(-180, 40, 0.25, 0.25),
                       crs='EPSG:4326') as dst:
        dst.write(diff.astype(np.float32), 1)
    
    with rasterio.open(os.path.join(output_path, "ChypSuitability1993-2022mean_MLDaver.tif"), 'w',
                       driver='GTiff', dtype=rasterio.float32,
                       height=mean_all.shape[0], width=mean_all.shape[1], count=1,
                       transform=from_origin(-180, 40, 0.25, 0.25),
                       crs='EPSG:4326') as dst:
        dst.write(mean_all.astype(np.float32), 1)
    
    # 确保图像输出目录存在
    os.makedirs('/home/yytan/Figures/Chyp-supp-MLDaver', exist_ok=True)
    
    # 绘制图像 - 保持原有代码不变
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=30)})
    
    # 画圆形边界Path
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    
    axes[0].set_extent([-180, 180, 55, 90], ccrs.PlateCarree())
    img1 = axes[0].imshow(mean_all[:-60, :], transform=ccrs.PlateCarree(), extent=[-180, 180, 55, 90],
                          origin='upper', cmap='jet', vmin=0, vmax=1, zorder=1)
    axes[0].set_title('Mean Suitability 1993-2022', pad=30)
    
    axes[0].add_feature(cfeature.LAND, facecolor="gray", zorder=2)
    axes[0].add_feature(cfeature.COASTLINE, edgecolor="black", zorder=3)
    axes[0].add_feature(cfeature.BORDERS, linestyle='--', zorder=3)
    axes[0].set_boundary(circle, transform=axes[0].transAxes)
    gl0 = axes[0].gridlines(draw_labels=True)
    gl0.ylocator = mticker.FixedLocator(range(60, 91, 10))
    cbar1 = plt.colorbar(img1, ax=axes[0], orientation='horizontal', pad=0.05, fraction=0.05)
    cbar1.set_label('C. hyperboreus c1-c2 habitat Suitability')
    
    axes[1].set_extent([-180, 180, 55, 90], ccrs.PlateCarree())
    img2 = axes[1].imshow(diff[:-60, :], transform=ccrs.PlateCarree(), extent=[-180, 180, 55, 90],
                          origin='upper', cmap='bwr', vmin=-0.5, vmax=0.5, zorder=1)
    axes[1].set_title('Suitability Change (2008-2022 minus 1993-2007)', pad=30)
    axes[1].add_feature(cfeature.LAND, facecolor="gray", zorder=2)
    axes[1].add_feature(cfeature.COASTLINE, edgecolor="black", zorder=3)
    axes[1].add_feature(cfeature.BORDERS, linestyle='--', zorder=3)
    axes[1].set_boundary(circle, transform=axes[1].transAxes)
    gl1 = axes[1].gridlines(draw_labels=True)
    gl1.ylocator = mticker.FixedLocator(range(60, 91, 10))
    cbar2 = plt.colorbar(img2, ax=axes[1], orientation='horizontal', pad=0.05, fraction=0.05)
    cbar2.set_label('C. hyperboreus c1-c2 habitat Suitability Change')
    
    plt.savefig('/home/yytan/Figures/Chyp-supp-MLDaver/result_2_MLDaver.jpg', dpi=500)  # 保存图片
    plt.close()
    
    print("Processing completed successfully!")
    print(f"Results saved to: {output_path}")
    print(f"Image saved to: /home/yytan/Figures/Chyp-supp-MLDaver/result_2_MLDaver.jpg")

def main():
    """主函数"""
    # 运行第一部分：MaxEnt预测和季节分析
    run_maxent_analysis()
    
    # 运行第二部分：年度均值计算和差异分析
    calculate_year_means()
    
    print("=" * 60)
    print("所有分析完成！")
    print("=" * 60)

if __name__ == '__main__':
    main() 
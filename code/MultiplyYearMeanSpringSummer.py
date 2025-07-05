# -*- coding: utf-8 -*-
"""
Created on ...
@author: ...
"""
import os
import glob
import numpy as np
import rasterio
from rasterio.transform import from_origin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import re

input_path = "/home/yytan/Maxent/Cgla-supp/"
output_path = "/home/yytan/Maxent/Cgla-supp/"

# 只遍历4位数字年份文件夹
ss_tif_files = []
for subdir in os.listdir(input_path):
    if re.fullmatch(r"\d{4}", subdir):
        year_folder = os.path.join(input_path, subdir)
        ss_tif_files.extend(sorted(glob.glob(os.path.join(year_folder, "CglaSuitability*.tif"))))

group1 = []  # 1993-2007
group2 = []  # 2008-2022

for tif in ss_tif_files:
    basename = os.path.basename(tif)
    date_str = basename.replace("CglaSuitability", "").replace(".tif", "")
    year = int(date_str[:4])
    month = int(date_str[4:6])
    if month < 4 or month > 9:
        continue  # 只保留4-9月
    if 1993 <= year <= 2007:
        group1.append(tif)
    elif 2008 <= year <= 2022:
        group2.append(tif)

def mean_rasters(file_list):
    if not file_list:
        raise ValueError("输入的文件列表为空，请检查分组条件和文件路径！")
    arr_list = []
    for f in file_list:
        with rasterio.open(f) as src:
            arr = src.read(1)
            arr = np.where((arr >= 0) & (arr <= 1), arr, np.nan)
            arr_list.append(arr)
    mean_arr = np.nanmean(arr_list, axis=0)
    return mean_arr

# 计算均值
mean1 = mean_rasters(group1)
mean2 = mean_rasters(group2)
diff = mean2 - mean1

# 计算1993-2022年所有影像的平均
all_group = group1 + group2
mean_all = mean_rasters(all_group)

# 输出spring_summer后缀tif
for arr, fname in zip(
    [mean1, mean2, diff, mean_all],
    ["CglaSuitability1993-2007mean_spring_summer.tif", "CglaSuitability2008-2022mean_spring_summer.tif", "CglaSuitability2008-2022minus1993-2007_spring_summer.tif", "CglaSuitability1993-2022mean_spring_summer.tif"]):
    with rasterio.open(os.path.join(output_path, fname), 'w',
                       driver='GTiff', dtype=rasterio.float32,
                       height=arr.shape[0], width=arr.shape[1], count=1,
                       transform=from_origin(-180, 40, 0.25, 0.25),
                       crs='EPSG:4326') as dst:
        dst.write(arr.astype(np.float32), 1)

# 读取spring_summer平均和差值影像用于绘图
with rasterio.open(os.path.join(output_path, "CglaSuitability1993-2022mean_spring_summer.tif")) as src:
    mean_all_img = src.read(1)
with rasterio.open(os.path.join(output_path, "CglaSuitability2008-2022minus1993-2007_spring_summer.tif")) as src:
    diff_img = src.read(1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=30)})

# 画圆形边界Path
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

axes[0].set_extent([-180, 180, 55, 90], ccrs.PlateCarree())
img1 = axes[0].imshow(mean_all_img[:-60, :], transform=ccrs.PlateCarree(), extent=[-180, 180, 55, 90],
                      origin='upper', cmap='jet', vmin=0, vmax=1, zorder=1)
axes[0].set_title('Mean Suitability 1993-2022 (April-September)', pad=30)
axes[0].add_feature(cfeature.LAND, facecolor="gray", zorder=2)
axes[0].add_feature(cfeature.COASTLINE, edgecolor="black", zorder=3)
axes[0].add_feature(cfeature.BORDERS, linestyle='--', zorder=3)
axes[0].set_boundary(circle, transform=axes[0].transAxes)
gl0 = axes[0].gridlines(draw_labels=True)
gl0.ylocator = mticker.FixedLocator(range(60, 91, 10))
cbar1 = plt.colorbar(img1, ax=axes[0], orientation='horizontal', pad=0.05, fraction=0.05)
cbar1.set_label('C. glacialis c1-c2 habitat Suitability')

axes[1].set_extent([-180, 180, 55, 90], ccrs.PlateCarree())
img2 = axes[1].imshow(diff_img[:-60, :], transform=ccrs.PlateCarree(), extent=[-180, 180, 55, 90],
                      origin='upper', cmap='bwr', vmin=-0.5, vmax=0.5, zorder=1)
axes[1].set_title('Suitability Change (2008-2022 minus 1993-2007, April-September)', pad=30)
axes[1].add_feature(cfeature.LAND, facecolor="gray", zorder=2)
axes[1].add_feature(cfeature.COASTLINE, edgecolor="black", zorder=3)
axes[1].add_feature(cfeature.BORDERS, linestyle='--', zorder=3)
axes[1].set_boundary(circle, transform=axes[1].transAxes)
gl1 = axes[1].gridlines(draw_labels=True)
gl1.ylocator = mticker.FixedLocator(range(60, 91, 10))
cbar2 = plt.colorbar(img2, ax=axes[1], orientation='horizontal', pad=0.05, fraction=0.05)
cbar2.set_label('C. glacialis c1-c2 habitat Suitability Change')

plt.savefig('/home/yytan/Figures/Cgla-supp/result_spring_summer.jpg', dpi=500)
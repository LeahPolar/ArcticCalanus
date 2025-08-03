# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 00:35:37 2025

@author: LeahPolar
"""

import os
import glob
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
from matplotlib import gridspec
import matplotlib.path as mpath
import re
from rasterio.transform import from_origin

# 更新输入和输出路径为MLDaver版本
input_path = "/home/yytan/Maxent/Cgla-supp-MLDaver/"
output_path = "/home/yytan/Maxent/Cgla-supp-MLDaver/"

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

tif_files = []
for subdir in os.listdir(input_path):
    if re.fullmatch(r"\d{4}", subdir):  # 只匹配4位数字的文件夹
        year_folder = os.path.join(input_path, subdir)
        tif_files.extend(sorted(glob.glob(os.path.join(year_folder, "CglaSuitability*.tif"))))

print(f"Found {len(tif_files)} TIF files")

group1 = []  # 1993-2007
group2 = []  # 2008-2022

for tif in tif_files:
    # 提取日期
    basename = os.path.basename(tif)
    date_str = basename.replace("CglaSuitability", "").replace(".tif", "")
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
with rasterio.open(os.path.join(output_path, "CglaSuitability1993-2007mean_MLDaver.tif"), 'w',
                   driver='GTiff', dtype=rasterio.float32,
                   height=mean1.shape[0], width=mean1.shape[1], count=1,
                   transform=from_origin(-180, 40, 0.25, 0.25),
                   crs='EPSG:4326') as dst:
    dst.write(mean1.astype(np.float32), 1)

with rasterio.open(os.path.join(output_path, "CglaSuitability2008-2022mean_MLDaver.tif"), 'w',
                   driver='GTiff', dtype=rasterio.float32,
                   height=mean2.shape[0], width=mean2.shape[1], count=1,
                   transform=from_origin(-180, 40, 0.25, 0.25),
                   crs='EPSG:4326') as dst:
    dst.write(mean2.astype(np.float32), 1)

with rasterio.open(os.path.join(output_path, "CglaSuitability2008-2022minus1993-2007_MLDaver.tif"), 'w',
                   driver='GTiff', dtype=rasterio.float32,
                   height=diff.shape[0], width=diff.shape[1], count=1,
                   transform=from_origin(-180, 40, 0.25, 0.25),
                   crs='EPSG:4326') as dst:
    dst.write(diff.astype(np.float32), 1)

with rasterio.open(os.path.join(output_path, "CglaSuitability1993-2022mean_MLDaver.tif"), 'w',
                   driver='GTiff', dtype=rasterio.float32,
                   height=mean_all.shape[0], width=mean_all.shape[1], count=1,
                   transform=from_origin(-180, 40, 0.25, 0.25),
                   crs='EPSG:4326') as dst:
    dst.write(mean_all.astype(np.float32), 1)

# 确保图像输出目录存在
os.makedirs('/home/yytan/Figures/Cgla-supp-MLDaver', exist_ok=True)

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
cbar1.set_label('C. glacialis c1-c2 habitat Suitability')

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
cbar2.set_label('C. glacialis c1-c2 habitat Suitability Change')

plt.savefig('/home/yytan/Figures/Cgla-supp-MLDaver/result_2_MLDaver.jpg', dpi=500)  # 保存图片
plt.close()

print("Processing completed successfully!")
print(f"Results saved to: {output_path}")
print(f"Image saved to: /home/yytan/Figures/Cgla-supp-MLDaver/result_2_MLDaver.jpg") 
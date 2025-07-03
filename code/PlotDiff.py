# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 19:41:51 2025

@author: LeahPolar
"""

import os
import glob
import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
from matplotlib import gridspec
import matplotlib.path as mpath
import re
from rasterio.transform import from_origin

# 读取第一个tif文件
tif1_path = r'D:\0-Calanus\2-Figures\20250703\CglaSuitability1993-2022mean.tif'
with rasterio.open(tif1_path) as src1:
    data1 = src1.read(1)  # 读取第一波段
    profile1 = src1.profile

# 读取第二个tif文件
tif2_path = r'D:\0-Calanus\2-Figures\20250703\CglaSuitability2008-2022minus1993-2007.tif'
with rasterio.open(tif2_path) as src2:
    data2 = src2.read(1)
    profile2 = src2.profile

fig, axes = plt.subplots(
    1, 2, figsize=(12, 6),
    subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=30)}
)

# 画圆形边界Path
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

axes[0].set_extent([-180, 180, 55, 90], ccrs.PlateCarree())
img1 = axes[0].imshow(data1[:-60, :], transform=ccrs.PlateCarree(), extent=[-180, 180, 55, 90],
                      origin='upper', cmap='jet', vmin=0, vmax=1, zorder=1)
axes[0].set_title('Mean Suitability 1993-2022', pad=30)
axes[0].add_feature(cfeature.LAND, facecolor="gray", zorder=2)
axes[0].add_feature(cfeature.COASTLINE, edgecolor="black", zorder=3)
axes[0].add_feature(cfeature.BORDERS, linestyle='--', zorder=3)
axes[0].set_boundary(circle, transform=axes[0].transAxes)
gl0 = axes[0].gridlines(draw_labels=True)
gl0.ylocator = mticker.FixedLocator(range(60, 91, 10))
cbar1 = plt.colorbar(img1, ax=axes[0], orientation='horizontal', pad=0.12, fraction=0.05)
cbar1.set_label('C. glacialis c1-c2 habitat Suitability')

axes[1].set_extent([-180, 180, 55, 90], ccrs.PlateCarree())
img2 = axes[1].imshow(data2[:-60, :], transform=ccrs.PlateCarree(), extent=[-180, 180, 55, 90],
                      origin='upper', cmap='bwr', vmin=-0.5, vmax=0.5, zorder=1)
axes[1].set_title('Suitability Change (2008-2022 minus 1993-2007)', pad=30)
axes[1].add_feature(cfeature.LAND, facecolor="gray", zorder=2)
axes[1].add_feature(cfeature.COASTLINE, edgecolor="black", zorder=3)
axes[1].add_feature(cfeature.BORDERS, linestyle='--', zorder=3)
axes[1].set_boundary(circle, transform=axes[1].transAxes)
gl1 = axes[1].gridlines(draw_labels=True)
gl1.ylocator = mticker.FixedLocator(range(60, 91, 10))
cbar2 = plt.colorbar(img2, ax=axes[1], orientation='horizontal', pad=0.12, fraction=0.05)
cbar2.set_label('C. glacialis c1-c2 habitat Suitability Change')

plt.savefig('D:/0-Calanus/2-Figures/20250703/result.jpg', dpi=500)  # 保存图片

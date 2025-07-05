# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:02:20 2025

@author: Leah
"""

import pandas as pd
import numpy as np
# 1. 读取两个 CSV 文件
df1 = pd.read_csv("FinalMatchuped3-CH.csv")  # 主表
df2 = pd.read_csv("Chyp_c1c2.csv")      # 副表（包含 Cgla_occur 列）
df2 = df2[df2["Chyp_c1c2"]>=0]
df2["Chyp_occur"] = df2["Chyp_c1c2"].apply(lambda x:1 if x>0 else 0)
# 2. 选择要合并的列
common_cols = ["lat", "lon", "year", "month", "day", "SIC", "CHL", "SST", "Sal", 
               "u", "v", "npp", "nitrat", "SIT", "mlotst_glor"]

# 3. 合并数据
# 从 df2 提取需要的列（包括 Cgla_occur）
df2_selected = df2[common_cols + ["Chyp_occur"]]

# 在 df1 中添加 Cgla_occur 列并赋值为 1
df1["Chyp_occur"] = 1

# 合并两个 DataFrame
merged_df = pd.concat([df1[common_cols + ["Chyp_occur"]], df2_selected], ignore_index=True)

# 检查相同位置和时间的行中Chyp_occur列是否一致
print("检查相同位置和时间的行中Chyp_occur列是否一致...")
key_cols = ["lat", "lon", "year", "month", "day"]

# 找出重复的行（基于位置和时间）
duplicates = merged_df[merged_df.duplicated(subset=key_cols, keep=False)]

if len(duplicates) > 0:
    print(f"发现 {len(duplicates)} 行有重复的位置和时间")
    
    # 按位置和时间分组检查Chyp_occur是否一致
    inconsistent_groups = []
    
    for name, group in duplicates.groupby(key_cols):
        if group['Chyp_occur'].nunique() > 1:
            # 找到不一致的行号
            inconsistent_rows = group.index.tolist()
            inconsistent_groups.append({
                'location_time': name,
                'row_indices': inconsistent_rows,
                'Chyp_occur_values': group['Chyp_occur'].tolist()
            })
    
    if inconsistent_groups:
        print(f"发现 {len(inconsistent_groups)} 组不一致的数据:")
        for i, group in enumerate(inconsistent_groups):
            print(f"  组 {i+1}:")
            print(f"    位置和时间: {group['location_time']}")
            print(f"    行号: {group['row_indices']}")
            print(f"    Chyp_occur值: {group['Chyp_occur_values']}")
            print()
    else:
        print("所有重复的位置和时间中，Chyp_occur列都是一致的")
else:
    print("没有发现重复的位置和时间")

merged_df = merged_df.drop_duplicates(subset=["lat", "lon", "year", "month", "day"], keep="first")
# 4. 保存合并后的数据
merged_df = merged_df.dropna()

merged_df.to_csv("Merged_Final_Chyp.csv", index=False)

print("合并完成，已保存为 Merged_Final_Chyp.csv")

df3 = pd.read_csv("FinalMatchuped6-MOSAiC.csv")

def sum_non_nan(row):
    if row.isna().all():
        return np.nan
    return row.sum()


df3["Chyp_c1c2"] = df3[['C. hyperboreus c1 _abunm2', 'C. hyperboreus c2 _abunm2']].apply(sum_non_nan, axis=1)
df3 = df3[df3["Chyp_c1c2"]>=0]
df3["Chyp_occur"] = df3["Chyp_c1c2"].apply(lambda x:1 if x>0 else 0)

df3 = df3.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})

final_columns = ["lat", "lon", "year", "month", "day", "SIC", "CHL", "SST", "Sal", 
               "u", "v", "npp", "nitrat", "SIT", "mlotst_glor", 'Chyp_occur']
result = pd.concat([df3[final_columns],merged_df[final_columns]], ignore_index=True)

# 保存输出（可选）
result.to_csv('Merged_Final_Chyp1.csv', index=False)
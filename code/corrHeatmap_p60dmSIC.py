# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:18:24 2024

@author: Yiyang Tan
基于p60dmSIC表的相关性热图分析
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def create_correlation_heatmap(input_file, output_file, title):
    """
    为单个文件创建相关性热图
    """
    print(f"处理文件: {input_file}")
    
    # 读取数据
    df = pd.read_csv(input_file)
    print(f"  总记录数: {len(df)}")
    
    # 选择环境参数列 - 根据p60dmSIC表的结构调整
    # 包含原有的环境参数和新增的p60dmSIC
    columns = ['SIC', 'CHL', 'SST', 'Sal', 'nitrat', 'mlotst_glor', 'MLDaverCHL', 
               'MLDaverSal', 'MLDaverSST', 'p60dmSIC']
    
    # 只选择存在的列
    available_columns = [col for col in columns if col in df.columns]
    print(f"  可用列: {available_columns}")
    
    if len(available_columns) < 2:
        print(f"  错误：可用列数不足，无法计算相关性")
        return
    
    # 选择数据
    df_selected = df[available_columns]
    
    # 删除包含NaN的行
    df_clean = df_selected.dropna()
    print(f"  清理后记录数: {len(df_clean)}")
    
    if len(df_clean) < 10:
        print(f"  错误：清理后数据不足，无法进行可靠的相关性分析")
        return
    
    # 计算相关性矩阵
    corr = df_clean.corr()
    
    # 创建上三角掩码
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 设置matplotlib图形
    plt.figure(figsize=(12, 10))
    
    # 生成自定义发散色图
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # 绘制热图
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f",
                cbar_kws={"shrink": .8}, annot_kws={"size": 8})
    
    # 调整布局
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=500, bbox_inches='tight')
    print(f"  图片已保存到: {output_file}")
    plt.close()

def main():
    """
    主函数
    """
    # 输入文件夹
    input_dir = r"E:\ArcticCalanus\standardized"
    
    # 输出文件夹
    output_dir = r"E:\ECNU\20250803"
    
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 输入文件
    cgla_input = os.path.join(input_dir, "Cgla_c1c2_p60dmSIC.csv")
    chyp_input = os.path.join(input_dir, "Chyp_c1c2_p60dmSIC.csv")
    
    # 输出文件
    cgla_output = os.path.join(output_dir, "Cgla_correlation_heatmap_p60dmSIC.jpg")
    chyp_output = os.path.join(output_dir, "Chyp_correlation_heatmap_p60dmSIC.jpg")
    
    # 检查输入文件是否存在
    if not os.path.exists(cgla_input):
        print(f"错误：输入文件不存在: {cgla_input}")
        return
    
    if not os.path.exists(chyp_input):
        print(f"错误：输入文件不存在: {chyp_input}")
        return
    
    # 处理Cgla文件
    print("=" * 50)
    print("处理Cgla相关性热图")
    print("=" * 50)
    create_correlation_heatmap(cgla_input, cgla_output, "Cgla Environmental Variables Correlation (with p60dmSIC)")
    
    # 处理Chyp文件
    print("\n" + "=" * 50)
    print("处理Chyp相关性热图")
    print("=" * 50)
    create_correlation_heatmap(chyp_input, chyp_output, "Chyp Environmental Variables Correlation (with p60dmSIC)")
    
    print("\n" + "=" * 50)
    print("所有相关性热图分析完成！")
    print("=" * 50)
    print(f"Cgla结果: {cgla_output}")
    print(f"Chyp结果: {chyp_output}")

if __name__ == "__main__":
    main() 
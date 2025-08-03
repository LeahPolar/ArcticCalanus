# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:40:37 2025

@author: Leah
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("D:/0-Calanus/0-SurveyData/Chyp_c1c2.csv")

# Select the columns of interest
columns = ['SIC', 'CHL', 'SST', 'Sal', 'u', 'v', 'npp', 'nitrat', 'SIT', 
           'mlotst_glor', 'o2_surface', 'po4_surface', 'si_surface']
df = df[columns]

# Calculate correlation matrix
corr = df.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f",
            cbar_kws={"shrink": .5}, annot_kws={"size": 8})

# Adjust layout and display
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("D:/0-Calanus/2-Figures/20250415/heatmap.jpg", dpi=500)
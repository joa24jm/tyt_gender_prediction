# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:49:28 2020

@author: joa24jm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

df = pd.read_excel('results/compare_distributions.xlsx')

age_comparison = df.iloc[1:13,7:]
age_comparison.columns = df.iloc[0, 7:]
x = np.arange(1, age_comparison.shape[0]+1)

#%%
# set figsize, i.e. set aspect ratio 1/2 = height/width
figsize = plt.figaspect(.5)

# create figure and subplots
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize=figsize)

# plot males
ax1.plot(x, age_comparison['male_all'].values, 
         color = 'yellowgreen', linestyle = 'dashed', label = 'all, n = 1871')
ax1.plot(x, age_comparison['male_sub_df'].values, 
         color = 'darkgreen', linestyle = 'solid', label = 'used for ML, n = 238')
ax1.set_title('Males, p = 0.354')
ax1.legend(loc = 'upper left')
ax1.set_xticks(x)
ax1.set_xticklabels(age_comparison['Decade of birth'], rotation = 90)
vals = ax1.get_yticks()
ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

# plot females
ax2.plot(x, age_comparison['female_all'].values, 
         color = 'lightsalmon', linestyle = 'dashed', label = 'all, n = 875')
ax2.plot(x, age_comparison['female_sub_df'].values, 
         color = 'saddlebrown', linestyle = 'solid', label = 'used for ML, n = 94')
ax2.legend(loc = 'upper left')
ax2.set_title('Females, p = 0.056')
ax2.set_xticks(x)
ax2.set_xticklabels(age_comparison['Decade of birth'], rotation = 90)

plt.tight_layout()

# save fig
plt.savefig('results/plots/age_density_comparsion2.svg')

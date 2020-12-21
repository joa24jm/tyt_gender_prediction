"""
Create boxplots per gendergroup for filled out questionnaires
"""

##% import modules
import pandas as pd
import numpy as np

#%% import dataframe
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')

#%% data for ax21
df_m = df[df['gender'] == 0]
df_f = df[df['gender']== 1]

males = df_m.user_id.value_counts()
females = df_f.user_id.value_counts()

#%% create boxplot
import matplotlib.pyplot as plt
data = [males, females]
import seaborn as sns
sns.set_theme(style="whitegrid")


fig, (ax1, ax2) = plt.subplots(1,2)
ax2.set_yscale('log')
flier_props = dict(marker="o", markersize=1)
bplot = ax2.boxplot(data, flierprops = flier_props, 
           labels = ['Male', 'Female'],
           autorange = True,
           showmeans = True,
           meanline = True,
           patch_artist=True,
           medianprops = {'linewidth':0},
           meanprops = {'linewidth': 1,
                        'linestyle': '--',
                        'color': 'red'})
ax2.set_ylabel('Number of filled out daily questionnaires')

# color boxes
colors = ['yellowgreen', 'lightsalmon']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)



# title statistics
from scipy import stats
t, p = stats.ttest_ind(males, females, equal_var=False)
x = males
y = females
dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
title = f"t({int(dof)}) = {round(t,2)}, p < 0.05"
ax2.set_title(title)

ax2.plot([], [], linestyle = '--', linewidth=1, color='red', label='mean')
ax2.legend()

# plt.savefig('results/plots/filled_out_questionnaire_per_gender_boxplot.svg',
#             format = 'svg')

plt.show()





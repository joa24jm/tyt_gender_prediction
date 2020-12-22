# =============================================================================
# How many questionnaires did each user fill out per gender?
# =============================================================================

#%% import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#%% import df
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')

#%% create crosstabe to find out how many questionnaires per gender and user have been filled out
bins = [0, 1, 2, 3, 10, 100, 1000, 4000]
filled_out_bins = pd.cut(df.user_id.value_counts(), bins = bins)
crosstab = pd.crosstab(filled_out_bins, df.gender, normalize = False, colnames= ['Gender'])

#%% get pvalue
# def get_t_test(df):
number_of_filled_out_questionnaires_male = df[df['gender'] == 0].shape[0]
n_male = len(df[df['gender'] == 0].user_id.unique())
x1_bar = number_of_filled_out_questionnaires_male/n_male

number_of_filled_out_questionnaires_female = df[df['gender'] == 1].shape[0]
n_female = len(df[df['gender'] == 1].user_id.unique())
x2_bar = number_of_filled_out_questionnaires_female/n_female

s2_male = sum([(val - x1_bar)**2 for val in df[df['gender']==0].user_id.value_counts().values])/(n_male-1)
s2_female = sum([(val - x1_bar)**2 for val in df[df['gender']==1].user_id.value_counts().values])/(n_female-1)

s_p = (((n_male-1)*s2_male + (n_female-1)*s2_female)/(n_male + n_female -2))**0.5

t_value = (x1_bar - x2_bar) / (s_p*((1/n_male)+(1/n_female))**0.5)

from scipy.stats import t
alpha = 0.05
t.ppf(1 - alpha, df=n_male + n_female -2)

crosstab2 = pd.crosstab(df.gender,df.user_id.value_counts(), normalize = False, colnames= ['Gender'], rownames=['Number of filled out questionnaires'])

from scipy.stats import ttest_ind

statistic, pvalue = ttest_ind(crosstab2.iloc[0,:].values, crosstab2.iloc[1,:].values)
d_f = crosstab2.shape[0] - 1

title = f't(2740) = {round(t_value,2)}, pval < 0.05'
    
#%% visualize

sns.set_theme(style="whitegrid")



labels = list(crosstab.index)
men = crosstab[0].values
women = crosstab[1].values

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, (ax1, ax2) = plt.subplots(1,2)
rects1 = ax1.bar(x - width/2, men, width, label='Males', color = 'yellowgreen')
rects2 = ax1.bar(x + width/2, women, width, label='Females', color = 'lightsalmon')

# Add some text for labels, title and custom x-ax1is tick labels, etc.
ax1.set_ylabel('Number of individuals')
#ax1.set_title('How many follow-up questionnaires \n did each individual fill out?')
ax1.set_xlabel('Number of filled out daily questionnaires')
ax1.set_xticks(x)
ax1.set_yticks(range(0, 700, 100))
ax1.set_xticklabels(labels, rotation = 45)
# ax1.set_title(title)
ax1.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax1.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)


df_m = df[df['gender'] == 0]
df_f = df[df['gender']== 1]

males = df_m.user_id.value_counts()
females = df_f.user_id.value_counts()



data = [males, females]


ax2.set_yscale('log')
flier_props = dict(marker="o", markersize=1)
bplot = ax2.boxplot(data, flierprops = flier_props, 
           labels = ['Males', 'Females'],
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

fig.tight_layout()

plt.savefig('results/plots/filled_out_questionnaire_per_user_and_gender.svg',
            format = 'svg')

plt.show()





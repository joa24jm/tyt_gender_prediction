# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:39:02 2020

@author: Johannes Allgaier

Julius Maximilian University of Würzburg
Am Schwarzenberg 15 / A15.4.311
97078 Würzburg
Germany

Phone | +49 931 201 46407
E-Mail | johannes.allgaier@uni-wuerzburg.de


Purpose of this file
Check Bias for df with feature question8_5, i.e.
compare the density of df_q8_5 with df
"""

# import packages
import pandas as pd
import modules as m

# read in df
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')
answers = pd.read_csv('codebook/exports/answers.csv', 
                      na_values = '??.??.????')

# feature list
cols = df.columns.tolist()[0:8] + ['question8_5'] + ['gender']

# create a sub_df with cols
sub_df = df[cols]

# drop NULL values
sub_df = sub_df.dropna(axis='index', how = 'any')

# equal split 50% male 50% female
sub_df = m.equal_split(sub_df)

# create a copy of the original df with equal split male female
df2 = m.equal_split(df)

# get lists with the users from both dfs
sub_df_users = sub_df['user_id'].unique().tolist()
df2_users = df2['user_id'].unique().tolist()

# answers drop 
answers = answers[answers['question_id']==4].dropna(how='any', subset=['answer'])

# select users from user lists
df2_answers = answers[answers['user_id'].isin(df2_users)]
sub_df_answers = answers[answers['user_id'].isin(sub_df_users)]

df_q4 = df2_answers[df2_answers['question_id']==4]
sub_df_q4 = sub_df_answers[sub_df_answers['question_id']==4]


df_dates = [int(df_q4['answer'].iloc[row][6:10]) for row in range(df_q4.shape[0])]
sub_df_dates = [int(sub_df_q4['answer'].iloc[row][6:10]) for row in range(sub_df_q4.shape[0])]

# plot density
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
plt.rc('font', size=10)  

n_bins = np.arange(1900, 2020, 5)

fig, ax = plt.subplots(nrows=1, ncols=1)

ax.hist(df_dates, bins = n_bins, density = True, histtype='step', edgecolor = '#581845', 
        label = 'All users', lw=1)
ax.hist(sub_df_dates, bins = n_bins, density = True, histtype='step', edgecolor = '#52be80', 
        linestyle = '--', label = 'Users used for ML approach', lw=1)

ax.set_xlabel('Year of birth')
ax.set_ylabel('Density')
ax.set_title('Comparsion of density for all users and the subset users')

plt.legend(bbox_to_anchor=(-0.14,-0.05), loc="upper left")

plt.tight_layout()

plt.savefig('results/plots/age_density_comparsion.svg', 
            format = 'svg',
            bbox_inches = "tight")

plt.show()

# =============================================================================
# Left handed, right handed, both sides comparsion
# =============================================================================

answers = pd.read_csv('codebook/exports/answers.csv', 
                      na_values = '??.??.????')

# answers drop 
answers = answers[answers['question_id']==6].dropna(how='any', subset=['answer'])

# select users from user lists
df2_answers = answers[answers['user_id'].isin(df2_users)]
sub_df_answers = answers[answers['user_id'].isin(sub_df_users)]

df_q6 = df2_answers[df2_answers['question_id']==6]
sub_df_q6 = sub_df_answers[sub_df_answers['question_id']==6]

q6_sub = sub_df_q6['answer'].value_counts(normalize = True)
q6 = df_q6['answer'].value_counts(normalize = True)

# Export values to excel and plot them using excel
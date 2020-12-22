# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:46:41 2020

@author: joa24jm

Compare the distributions of the whole df df = ('data/df_q1_q8_0_q8_8.xlsx')
with a subset of q1, q2, ...., q7 + q8_5

"""

import pandas as pd
from scipy.stats import ks_2samp
import numpy as np

# check for age
def compare_distributions(data1, data2):
    """
    Compute the Kolmogorov-Smirnov statistic on 2 samples.
    This is a two-sided test for the null hypothesis 
    that 2 independent samples are drawn from the same continuous distribution
    """

    statistic, pvalue = ks_2samp(data1, data2)
    
    return {'statistic': statistic, 'pvalue':pvalue}

# import baseline questionnaire
bl = pd.read_excel('data/baseline.xlsx', index_col = 'Unnamed: 0')
bl['Date of Birth'] =  pd.to_datetime(bl['Date of Birth'], errors='coerce').dt.year

# import daily questionnaire
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx', index_col = 'user_id')

# create sub_df_users by gender
df_male = df[df['gender'] == 0]
df_female = df[df['gender'] == 1]
sub_df_male = df_male[df_male['question8_5'].notna()]
sub_df_female = df_female[df_female['question8_5'].notna()]
sub_df_users = {'Male':   list(sub_df_male.index.unique()),
                'Female': list(sub_df_female.index.unique())}

# get all users by gender
all_users = {'Male':   list(df[df['gender'] == 0].index.unique()),
             'Female': list(df[df['gender'] == 1].index.unique())}

# columns of interest for comparison


# group df by gender
bl_male = bl[bl['Gender'] == 'Male']
bl_female = bl[bl['Gender'] == 'Female']

     
# get value counts
for col in ['Handedness','Family history of tinnitus complaints']:
    for df, gender in zip([bl_male, bl_female], ['Male', 'Female']):
        
        # print and copy to excel
        print(f'{col}\t{gender}')
        print(f'all users: n =  {len(df.loc[all_users[gender],col])}')
        print(df.loc[all_users[gender],col].value_counts(normalize = True))
        print(f'sub_df_users: n = {len(df.loc[sub_df_users[gender],col])}')
        print(df.loc[sub_df_users[gender],col].value_counts(normalize = True))
        print('\n\n')

bl_male['Date of Birth'].hist()

# Age Distribution Male All Users
bins = np.arange(1900, 2030, 10)
series1 = bl_male.loc[sub_df_users['Male'], 'Date of Birth']
series2 = bl_male.loc[all_users['Male'], 'Date of Birth']
series3 = bl_female.loc[sub_df_users['Female'], 'Date of Birth']
series4 = bl_female.loc[all_users['Female'], 'Date of Birth']

# get age bins as indices
idxs = pd.cut(series1, bins).value_counts(normalize = True).sort_index().index

# define resulting df
age_result = pd.DataFrame(index = idxs,
                          columns=['male_sub_df', 'male_all', 'female_sub_df', 'female_all'])

# fill df with values
for s, col in zip([series1, series2, series3, series4], age_result.columns):
    age_result.loc[idxs, col] = pd.cut(s, bins).value_counts(normalize = True).sort_index()
    
statistics_male = compare_distributions(series1.values, series2.values)
statistics_female = compare_distributions(series3.values, series4.values)

# get age characteristics

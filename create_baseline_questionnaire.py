# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:57:06 2020

@author: joa24jm

create readable baseline questionnaire
"""

import pandas as pd
import modules as m
import numpy as np

# import dataframes
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')
bl = pd.read_excel('codebook/codebook_johannes.xlsx', sheet_name='answers')
codebook = pd.read_excel('codebook/codebook_johannes.xlsx', sheet_name='codebook')[['question_id', 'question / meaning']]

codebook = dict(zip(codebook['question_id'].values, codebook['question / meaning'].values))

# preprocess ages
ages_df['answer'].replace({'??.??.????':np.nan}, inplace = True)


# create a readable baseline df
questions_list = sorted(bl['question_id'].dropna().unique())

# list of users
users_ls = bl['user_id'].dropna().unique()
result = pd.DataFrame(index = users_ls)

# columns to merge on 
cols = ['answer', 'user_id']
#%%
for q in questions_list:
    # create filter
    filt = bl['question_id'] == q
    # create df with answers
    df = bl[filt][cols].set_index('user_id')
    # rename column for unique column names
    df.rename(columns={'answer': codebook[q]}, inplace = True)
    # merge with result
    result = result.merge(df, how='left', 
                          left_index = True, right_index = True)
    
# save result
result.to_excel('data/baseline.xlsx')



















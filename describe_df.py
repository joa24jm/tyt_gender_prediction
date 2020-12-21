# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:16:33 2020

@author: Johannes Allgaier
"""


# import modules
import pandas as pd


# import dataset
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')

# import and prepare codebook
cb = pd.read_excel('codebook/codebook_johannes.xlsx', sheet_name='codebook')
cb = cb[['question_id', 'question / meaning', 'datatype', 'Implementation']].iloc[0:16]
cb.rename(columns={'question_id':'label', 
                   'question / meaning': 'meaning'}, inplace = True)

# create a codebook dict for easier access
cb_dic = dict(zip(cb.label, cb.meaning))

cb_dat = dict(zip(cb.label, cb.datatype))
cb_dat['gender'] = 'binary'

# create df description
df_descr = df.describe().transpose()

# drop user_id for description
df_descr.drop(index='user_id', inplace = True)

# add scaling column
df_descr['scaling'] = None

for key in cb_dat.keys():
    df_descr.loc[key, 'scaling'] = cb_dat[key]
    

# add meaning and Implementation to descr
df_descr = pd.concat([df_descr, cb.set_index('label')[['meaning', 'Implementation']]], axis = 1)

# add information about gender
df_descr.loc['gender', 'meaning'] = '0 = Male, 1 = Female'
df_descr.loc['gender', 'Implementation'] = 'SingleChoice'

# add information about q8_3
df_descr.loc['question8_3', ['scaling', 'Implementation', 'count', 'mean', 'std']] = 'NULL'

# drop quantiles
df_descr.drop(columns = ['min', '25%', '50%', '75%', 'max'], inplace = True)

# change order of columns
columnsTitles = ['meaning', 'scaling', 'Implementation', 'count', 'mean', 'std']
df_descr = df_descr.reindex(columns=columnsTitles)

# round cols
df_descr = df_descr.astype({'count':'int'})
df_descr[['mean', 'std']] = df_descr[['mean', 'std']].round(2)

# save description to csv
df_descr.to_excel('results\df_description.xlsx')

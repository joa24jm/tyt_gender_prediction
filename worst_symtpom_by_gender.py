# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:44:18 2020

@author: Johannes Allgaier
"""

import pandas as pd

# read in df
df = pd.read_excel('data/baseline.xlsx')

# drop trash values
trash_value = "Because of the tinnitus I am more irritable with my family, friends and colleagues.I find it harder to relax because of the tinnitus.I have strong worries because of the tinnitus.Because of the tinnitus it is hard for me to get to sleep.Because of the tinnitus it is difficult to concentrate.Because of the tinnitus I am more sensitive to environmental noises.Because of the tinnitus it is difficult to follow a conversation, a piece of music or a film.I am feeling depressed because of the tinnitus.I don't have any of these symptoms."
rows_to_drop = list(df[df['Please tick only the worst symptom'] == trash_value].index)
df = df.drop(rows_to_drop)


# group by gender
gender_grp = df.groupby('Gender')

# pd crosstab
worst_symptom_by_gender = pd.crosstab(df['Please tick only the worst symptom'], df['Gender'],
            normalize = 'columns')

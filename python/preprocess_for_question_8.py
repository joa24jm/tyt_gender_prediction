# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:37:21 2020

@author: Johannes Allgaier

Julius Maximilian University of Würzburg
Am Schwarzenberg 15 / A15.4.311
97078 Würzburg
Germany

Phone | +49 931 201 46407
E-Mail | johannes.allgaier@uni-wuerzburg.de


Purpose of this file
Create dataframes that also contains answers from question8, like so:

for i in answers.unique().tolist(): 
    SELECT user_id
    FROM answers
    WHERE answer = i

"""
# activate venv using 'conda activate venv_tyt'

# import required package
import pandas as pd
import modules as m
import numpy as np

# read in dataframes
# define missing values
# na_values = [-0.01, 'nan']
ans = pd.read_excel('codebook/codebook_johannes.xlsx',
                    sheet_name = 'answers')

ans['user_id'] = ans.user_id.astype('category')

dai = pd.read_excel('codebook/codebook_johannes.xlsx',
                    sheet_name = 'daily_questions_answers')

dai['user_id'] = dai.user_id.astype('category')

# define target cols for daily questionaire
target_cols = ['user_id',   'question1', 'question2', 'question3', 
               'question4', 'question5', 'question6', 'question7',
               'question8']

dai = dai[target_cols]
dai = dai[dai['user_id'].notna()]

# replace wrong values
dai['question4'] = m.replace_wrong_values(dai['question4'])
dai['question5'] = m.replace_wrong_values(dai['question5'])

# Missing value treatment
# note that this function automatically drops missing user_ids (998 missing)
dai = m.missing_value_treatment(dai, target_cols[1:])

# select question 24: 'Please tick only the worst symptom'
filt = ans['question_id'] == 24
q24 = ans[filt]

# find all possible answers except the last one (last on is a bug answer that contains all answers)
answers = q24['answer'].unique().tolist()[:-1]

# select question 5: 'Gender'
filt2 = ans['question_id'] == 5 
q5 = ans[filt2][['user_id', 'answer']]

# set up a dict to save merged dfs
dfs = {}

# create different datasets per question8
"""
for i, answer in enumerate(answers):

    # select all users that answered with i
    filt3 = q24['answer'] == answer
    q24_ai = q24[filt3]['user_id']
    
    # merge with q24_ai (daily answers groubed by answers from first questionaire)
    merged = dai.merge(q24_ai, on = 'user_id')
    merged = merged[merged['user_id'].notna()]
    
    # merge with q5 (Gender)
    merged = merged.merge(q5, on = 'user_id', how = 'left')
    merged = merged.rename(columns={'answer':'gender'})
    
    # append merged to dict 
    dfs[f'ans{i}'] = merged
    
    # save to excel
    merged.to_excel(f'data/df_q1_q8_answer_{i}.xlsx', index = False)
"""

def create_merged_df(daily_questionaire, q5_gender, q24_worst_symptom):
    """
        Parameters
    ----------
    dai : DataFrame
        Contains q1, q2, ..., q8
    q5 : DataFrame
        Contains gender and user_id
    q24 : DataFrame
        Contains the worst symptom per user

    Returns
    -------
    Merged dataframe with cols
    user_id, q1, ..., q7, q8_0, q8_1, ..., q8_8, gender

    """
    # define list of all columns
    cols = daily_questionaire.columns.tolist()
    for i in range(len(answers)):
        col = f'question8_{i}'
        cols.append(col)
    cols.append('gender')
    
    # 
    for i, answer in enumerate(answers):
        # select all users that answered with i
        filt3 = q24_worst_symptom['answer'] == answer
        # get a list of all user_ids that answered with answer i
        q24_ai = q24_worst_symptom[filt3]['user_id'].values.tolist()
    
        daily_questionaire[f'question8_{i}'] = daily_questionaire.loc[daily_questionaire['user_id'].isin(q24_ai)]['question8']
        
    return dai

dai = create_merged_df(dai, q5, q24)

# drop column question8
dai = dai.drop(['question8'], axis=1)

dai = dai.merge(q5, how = 'inner', on='user_id')    
dai.rename(columns={'answer':'gender'}, inplace = True)

# replace values
# replace boolean values
dai = dai.replace(to_replace = {True:1, False:0,
                              'Male':0, 'Female': 1})

# replace wrong values
dai['question4'] = m.replace_wrong_values(dai['question4'])
dai['question5'] = m.replace_wrong_values(dai['question5'])

# replace wrong values with NULL as this question was never given to the user
# It's not clear why question8_3 has values - do not drop.
# dai['question8_3'] = np.nan

# remove -0.01 values
for col in dai.columns:
    dai = dai.drop(dai[dai[col] == -0.01].index)

# summaraize dataframe into description
dai_description = dai.describe()

# save dataframe and description to excel
dai.to_excel('data/df_q1_q8_0_q8_8.xlsx', index = False) 
dai_description.to_excel('data/df_q1_q8_0_q8_8_description.xlsx', index = False) 
    
    
    
    
    
    
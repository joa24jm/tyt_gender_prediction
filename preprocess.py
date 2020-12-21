# =============================================================================
# # -*- coding: utf-8 -*-
# Created on Thu Jun 18 11:20:13 2020
# 
# @author: joa24jm
# 
# Project path
# # cd C:\Users\joa24jm\Dropbox\20-06-17_TYT
# =============================================================================

# =============================================================================
# Step 1: import required datasets and join on user_id
# =============================================================================
import pandas as pd
import modules as m

# daily questionaire, used for X
dq = pd.read_excel('codebook/codebook_johannes.xlsx', 
                   sheet_name = 'daily_questions_answers')

# only keep users where user_id is not False
filt = dq['user_id'].isnull() == False
dq = dq[filt]

# drop -0.01 values
dq = dq.drop(dq[dq['question3'] == -0.01].index)
dq = dq.drop(dq[dq['question4'] == -0.01].index)
dq = dq.drop(dq[dq['question6'] == -0.01].index)

# =============================================================================
# Step 2: Clean up your dataframe
# =============================================================================
columns =  ['id','user_id','question1','question2','question3','question4',
            'question5','question6','question7']
dq = m.missing_value_treatment(dq, columns)

# =============================================================================
# Step 3: Replace iOS and Android values for question4 and question5
# =============================================================================
dq['question4'] = m.replace_wrong_values(dq['question4'])
dq['question5'] = m.replace_wrong_values(dq['question5'])

# =============================================================================
# Step 4: Replace iOS and Android values for question4 and question5
# =============================================================================
# answers questionaire, used for y
aq = pd.read_excel('codebook/codebook_johannes.xlsx',
                   sheet_name = 'answers')

# creat a filter to get the gender
filt = aq['question_id'] == 5

# get a table that contains only question 5 (What is your gender?)
y = aq[filt][['user_id', 'answer']]
y['user_id'] = y['user_id'].astype(float)

# rename key column for more transparency
y = y.rename(columns={'answer':'gender'})

# =============================================================================
# Step 5: Add hand-made features per user
# =============================================================================
cols = ['user_id','question1','question2','question3','question4',
            'question5','question6','question7']

artificial_df = m.create_features(dq[cols])

# if n_filled_questionaires = 1, std is not calculatable, we set std = 0
artificial_df = artificial_df.fillna(0)

# =============================================================================
# Step 6: Merge artificial df with target
# =============================================================================

# merge X and y on user_id
df = artificial_df.merge(y, on ='user_id', how = 'inner')



# =============================================================================
# Pre-processing done, export df
# =============================================================================
df.to_excel('data\df_artificial.xlsx', index = False)




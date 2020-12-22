# -*- coding: utf-8 -*-
"""
Purpose of this file
Get baseline characteristics for all users that filled out the baseline 
questionnaire.
Note that these are not neccessarily the users that also filled out a 
follow-up questionnaire.


Created on Wed Jul 22 14:30:30 2020

@author: Johannes Allgaier

Julius Maximilian University of Würzburg
Am Schwarzenberg 15 / A15.4.311
97078 Würzburg
Germany

Phone | +49 931 201 46407
E-Mail | johannes.allgaier@uni-wuerzburg.de


Purpose of this file
Create baseline characteristics for gender, age, handedness
    
"""

# activate venv using 'conda activate name_of_venv'

# import required packages
import pandas as pd
import numpy as np

# import df
ans = pd.read_excel('C:/Users/joa24jm/Dropbox (University of Wuerzburg)/20-06-17_TYT/codebook/codebook_johannes.xlsx',
                    sheet_name = 'answers', na_values = '??.??.????')

# drop NULL ids
ans = ans[ans['user_id'].notnull()]

# select question 4: 'Age'
filt3 = ans['question_id'] == 4
q4 = ans[filt3][['user_id', 'answer']]
q4.rename(columns={'answer':'date_of_birth'}, inplace = True)

# select question 5: 'Gender'
filt2 = ans['question_id'] == 5 
q5 = ans[filt2][['user_id', 'answer']]
q5.rename(columns={'answer':'gender'}, inplace = True)

# select question 6: 'Handedness'
filt4 = ans['question_id'] == 6
q6 = ans[filt4][['user_id', 'answer']]
q6.rename(columns={'answer':'handedness'}, inplace = True)

# select question 6: 'Handedness'
filt7 = ans['question_id'] == 7
q7 = ans[filt7][['user_id', 'answer']]
q7.rename(columns={'answer':'history'}, inplace = True)

# merge three dataframes into one
merged = pd.merge(
            pd.merge(
                pd.merge(q4, q5, on='user_id',  how='outer'), 
                q6, on='user_id', how='outer'),
                    q7, on='user_id', how= 'outer')
merged['year'] = np.nan

# get the age of the patients
for row in range(merged.shape[0]):
    try:
        merged['year'][row] = int(merged['date_of_birth'][row][6:11])
    except:
        print('error in row: ', row)
        merged['year'][row] = np.nan
        
merged['year'] =  merged['year'].astype(float)

# create gender filters
male_filt = merged['gender'] == 'Male'
female_filt = merged['gender'] == 'Female'

# apply filters on columns
age_mean_male = 2020 - merged[male_filt].describe().loc['mean', 'year']
age_mean_female = 2020 - merged[female_filt].describe().loc['mean', 'year']

handedness_male = merged[male_filt]['handedness'].value_counts()
handedness_female = merged[female_filt]['handedness'].value_counts()

history_male = merged[male_filt]['history'].value_counts()


# =============================================================================
# How many questionnaires did each user fill out?'
# =============================================================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

dai = pd.read_excel('C:/Users/joa24jm/Dropbox (University of Wuerzburg)/20-06-17_TYT/codebook/codebook_johannes.xlsx',
                    sheet_name = 'daily_questions_answers', na_values = '??.??.????')


counts = dai['user_id'].value_counts().values



n, bins, patches = plt.hist(counts, bins = [1,2,5,10,100,1000, 10000], edgecolor = 'black')

labels = ['1', '2-4', '5-10', '11-99', '100-999', '>1000']
x = np.arange(len(labels)) 

plt.bar(x, height = n, color='#297db1')
plt.xticks(ticks=x, labels = labels)
plt.xlabel('Number of filled out questionnaires')
plt.ylabel('Number of users')
plt.title('How many questionnaires did each user fill out?')
# plt.savefig('results/plots/filled_out_questionnaire_per_user.svg', 
#             format = 'svg')
plt.show()
















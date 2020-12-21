# -*- coding: utf-8 -*-
"""
@author: Johannes Allgaier

Johannes Allgaier, MSc
Research Associate
Health informatics

Institute of Clinical Epidemiology and Biometry (ICE-B)

Julius Maximilian University of Würzburg
Am Schwarzenberg 15 / A15.4.311
97078 Würzburg
Germany

Phone | +49 931 201 46407
E-Mail | johannes.allgaier@uni-wuerzburg.de
Website: | https://www.med.uni-wuerzburg.de/epidemiologie/institut/mitarbeiter/team/johannes-allgaier/
"""


import pandas as pd
# import numpy as np

# read in df
ans = pd.read_excel('tables.xlsx', 
                    sheet_name = 'answers')

# create lists of questionaires and questions
questionaire_ids = ans['questionaire_id'].dropna().unique().tolist()
question_ids = ans['question_id'].dropna().unique().tolist()
cols = ['questionaire_id', 'question_id','question_text', 'answer']
codebook = pd.DataFrame(columns = cols)


# create filters while looping through lists
for questionaire in questionaire_ids:
    for question in question_ids:
        filt = (ans['questionaire_id']==questionaire) & (ans['question_id'] == question)
        question_texts = ans[filt]['question_text'].dropna().unique().tolist()
        answers = sorted(ans[filt]['answer'].dropna().unique().tolist())
        if len(answers) > 0:              
                to_append = pd.DataFrame(data = [[questionaire, question,question_texts[0], answers]],
                                     columns = cols)
                codebook = codebook.append(to_append, ignore_index = True)

# save codebooks to excel
codebook.to_excel('codebook.xlsx', index = False)


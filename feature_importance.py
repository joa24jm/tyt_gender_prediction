# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:58:29 2020

@author: Johannes Allgaier

Julius Maximilian University of Würzburg
Am Schwarzenberg 15 / A15.4.311
97078 Würzburg
Germany

Phone | +49 931 201 46407
E-Mail | johannes.allgaier@uni-wuerzburg.de


Purpose of this file
Get feature importance and feature correlation

"""

# imports 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# specify dtypes
dtypes = {'user_id':'category',
          'question1':'bool',
          'question2':'float64',
          'question3':'float64',
          'question4':'float64',
          'question5':'float64',
          'question6':'float64',
          'question7':'float',
          'question8_0':'category',
          'question8_1':'category',
          'question8_2':'category',
          'question8_3':'category',
          'question8_4':'category',
          'question8_5':'category',
          'question8_6':'category',
          'question8_7':'category',
          'question8_8':'category',
          'gender':'category'}

# import the dataset
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx', dtype = dtypes)

# split up male-female ratio to 50 percent
df_f = df[df['gender'] == 1].sample(frac=1) # all female users
df_m = df[df['gender'] == 0].sample(n=df_f.shape[0]) # n = n_female_users

df = df_m.append(df_f)

# get a list with all q8 questions
q8s = df.columns.tolist()[8:17]
qs_remain = df.columns.tolist()[1:8]

for q8 in q8s:
    
    # get a subset of the dataframe
    data = df[qs_remain + [q8] + ['gender']].dropna(how='any', axis = 'index')
    X = data[qs_remain + [q8]]
    y = data['gender']
    
    # split up the data
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size = .2,
                                                        random_state = 42)  
    
    rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                criterion='gini', max_depth=X.shape[1], 
                                max_features=X.shape[1],
                                max_leaf_nodes=None, max_samples=None,min_impurity_decrease=0.0, 
                                min_impurity_split=None, min_samples_leaf=1, 
                                min_samples_split=2,min_weight_fraction_leaf=0.0, 
                                n_estimators=100,n_jobs=None, oob_score=False,
                                random_state=42, verbose=0, warm_start=False)
    
    rf.fit(X_train, y_train)
    
    print(q8, '\t', round(f1_score(y_test, rf.predict(X_test)),4))
   
    result = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title(f'Permutation importances on test set including {q8}')
    fig.tight_layout()
    
    # plt.savefig(f'results/plots/permutation_importance_{q8}.svg', format='svg')
    plt.show()


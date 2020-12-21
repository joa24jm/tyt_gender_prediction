# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:39:49 2020

@author: Johannes Allgaier

Julius Maximilian University of Würzburg
Am Schwarzenberg 15 / A15.4.311
97078 Würzburg
Germany

Phone | +49 931 201 46407
E-Mail | johannes.allgaier@uni-wuerzburg.de


Purpose of this file
Variable importance for the dataset
Univariate Regression - importance measured as the accuracy
    
"""

# import required packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import modules as m
import matplotlib

# import dataset
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')

# import and prepare codebook
cb = pd.read_excel('codebook/codebook_johannes.xlsx', sheet_name='codebook')
cb = cb[['question_id', 'question / meaning']].iloc[0:17]
cb.rename(columns={'question_id':'label', 
                   'question / meaning': 'meaning'}, inplace = True)

# create a codebook dict for easier access
cb_dic = dict(zip(cb.label, cb.meaning))

# get all features of df
features = df.iloc[:,1:-1].columns.tolist()

# drop q8_3 as this is the FUP question to 'none of these worst symptoms'
features.remove('question8_3')

def calculate_permutation_importance(clf, X, y, n_repeats = 10, random_state = 0):
    """
    The estimator is required to be a fitted estimator. 
    X can be the data set used to train the estimator or a hold-out set. 
    The permutation importance of a feature is calculated as follows. 
    First, a baseline metric, defined by scoring, is evaluated on a (potentially different) dataset defined by the X. 
    Next, a feature column from the validation set is permuted and the metric is evaluated again. 
    The permutation importance is defined to be the difference between the baseline metric and metric from permutating the feature column.
    
    L. Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001. https://doi.org/10.1023/A:1010933404324

    Parameters
    ----------
    clf : Random forest classifier
    X : dataset with features
    y : target gender
    n_repeats : int, optional
        DESCRIPTION. The default is 10.
    random_state : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    [importances_mean, importances_std]

    """
    from sklearn.inspection import permutation_importance
    
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=0)
    
    result_arr = [result.importances_mean, result.importances_std]
    
    return result_arr
    

# define results dict
results = {}
for col in features:
    results[col] = []

# n_min for equal train-test chances 
test_size = .2
n_min = 2878
n_test = n_min * test_size

# set seed for replicable results
seed = 123456789

# iterate through features, calculate accuracy as measurement for feature importance
for col in features:  
    
    # drop na values
    sub_df = df[[col, 'gender']].dropna()  
  
    # get feature and corresponding target rows
    sub_df = m.equal_split(sub_df)
    
    X = sub_df[col]
    y = sub_df['gender'].loc[X.index]
    
    # reshape to (n_samples, 1)
    X = X.to_numpy().reshape(-1,1)    
    
    print(col, '\t', X.shape[0])
    
    #for i in range(10):
        
    # print(i)
    
    # define train-test-split
    X_train, X_test, y_train, y_test = m.train_test_set(X,y, seed)
    
    # instantiate and train a clf
    clf = RandomForestClassifier(random_state  = seed).fit(X_train, y_train)
    
    acc = calculate_permutation_importance(clf, X, y)
    
    # calculate y_pred
    # y_pred = clf.predict(X_test)
    
    # evaluate the clf
    # acc = round(accuracy_score(y_test, y_pred), 4)
    
    # append score to results list
    results[col].append(acc)
    
# plot the results using horizontal boxplots and a mapping to the questions
questions = list(cb_dic.keys())

# get a dataframe with results
data = pd.DataFrame(results.values(), index = list(results.keys()))

# sort data (descending by mean of accuracy values)
data_means = data.mean(axis=1).sort_values()

# instantiate pyplot object
plt.style.use('ggplot')

# set the default font
matplotlib.rcParams['font.sans-serif'] = "Computer Modern"

fig, ax = plt.subplots(nrows = 1, ncols = 1)


ax.barh(range(1,16), sorted(data_means))

ax.set_yticks(np.arange(1, len(data_means.index)+1))
ax.set_yticklabels([cb_dic[k] for k in data_means.index])
ax.set_xlim(left=.45, right = .6)
ax.set_xlabel('Accuracy of random forest classifier on test data')

ax.axvline(x=.5, linestyle = '--', lw=1, color = 'black', label = 'Random Guessing')

ax.set_title('Univariate feature importance for gender classification')
plt.legend(bbox_to_anchor=(-0.6,-0.1), loc="upper right")

plt.tight_layout()

# plt.savefig('results/plots/feature_importance_with_equal_splits.svg', 
#             format = 'svg',
#             bbox_inches = "tight")



plt.show()














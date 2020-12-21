# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:13:42 2020

@author: Johannes Allgaier

Julius Maximilian University of Würzburg
Am Schwarzenberg 15 / A15.4.311
97078 Würzburg
Germany

Phone | +49 931 201 46407
E-Mail | johannes.allgaier@uni-wuerzburg.de


Purpose of this file
Fine tune a classifier using 
['question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7', 'question8_5'] 
as features
    
"""

# =============================================================================
# Part I
# Train and evaluate a RF classifier on every possible combination of features
# =============================================================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import modules as m
from sklearn.metrics import accuracy_score

df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')

# extract feature names from columns
ls1 = df.columns.tolist()[1:8]  # q1, q2, ..., q7
ls2 = df.columns.tolist()[8:17] # q8_0, q8_1, ..., q8_7
ls2.remove('question8_3')

# get a list that contains a list with all possible combinations of columns
feature_lists = m.make_all_possible_feature_combinations(ls1, ls2)
columns = ['features', 'score', 'N']
result = pd.DataFrame(columns = columns)

for i, feature_list in enumerate(feature_lists):

    # Random Forest Classifier
    clf = RandomForestClassifier(random_state=1994) 
    
    # define features and target
    sub_df = df.loc[:, feature_list + ['gender']]
    
    # drop NULL values
    sub_df = sub_df.dropna(axis='index', how = 'any')
    
    # equal split 50% male 50% female
    sub_df = m.equal_split(sub_df)
    
    # define features and target
    X = sub_df.loc[:,feature_list]
    y = sub_df.iloc[:,-1]
    
    # split up train and test data
    X_train, X_test, y_train, y_test = m.train_test_set(X, y)
    
    # fit classifier to data
    clf.fit(X_train, y_train)
    
    # predict
    y_pred = clf.predict(X_test)
    
    # calculate accuracy
    score = round(accuracy_score(y_test, y_pred),4)
    
    to_append = pd.DataFrame(data = [[feature_list] + [score] + [sub_df.shape[0]]], columns = columns)
    result = result.append(to_append, ignore_index = True)
    
features = ls1 + ls2
for f in features: 
    result[f] = 0
    
for row in range(result.shape[0]):
    for f in result.loc[row, 'features']:
        result.loc[row, f] = 1
            
# result.to_excel('results/find_best_feature_combination.xlsx', index = False)
    
# =============================================================================
# Part II
# Hyperparameter Tuning of this Random Forest and those best features
# =============================================================================
from sklearn.model_selection import GridSearchCV

# find the features for the best score
idx = result['score'].idxmax(axis = 0)
best_features = result['features'].iloc[idx]

# grid search and cv for random forest classifier on best features
sub_df = df.loc[:,['question1','question2','question3', 'question4',
 'question5', 'question6', 'question7', 'question8_5','gender']]
# drop NULL values
sub_df = sub_df.dropna(axis='index', how = 'any')
# equal split 50% male 50% female
sub_df = m.equal_split(sub_df)

# get features and target
X = sub_df.loc[:,['question1','question2','question3', 'question4',
 'question5', 'question6', 'question7', 'question8_5']]
y = sub_df.iloc[:,-1]

# set up a default classifier
rfc = RandomForestClassifier(random_state=1994) 

# set up grid search parameters
parameters = {'bootstrap':                [True, False],
              'ccp_alpha':                [0.0], 
              'class_weight':             [None],
              'criterion':                ['gini', 'entropy'],
              'max_depth':                [None, 2, 5, 10, 20, 100],
              'max_features':             ['auto', 'sqrt', 'log2'],
              'max_leaf_nodes':           [None], 
              'max_samples':              [None],
              'min_impurity_decrease':    [0.0], 
              'min_impurity_split':       [None],
              'min_samples_leaf':         [1,2,10], 
              'min_samples_split':        [2],
              'min_weight_fraction_leaf': [0.0], 
              'n_estimators':             [1, 3, 5, 10, 100, 200,
                                           300, 500, 1000],
              'n_jobs':                   [None], 
              'oob_score':                [False], 
              'random_state':             [1994],
              'verbose':                  [0], 
              'warm_start':               [True, False]
              }

# clf = GridSearchCV(rfc, parameters)
# clf.fit(X, y)

# grid_search_results = pd.DataFrame(clf.cv_results_)

# Second approach
parameters2 = {'bootstrap':               [False],
              'ccp_alpha':                [0.0], 
              'class_weight':             [None],
              'criterion':                ['entropy'],
              'max_depth':                [10, 50, 90, 100, None],
              'max_features':             ['auto', 'sqrt'],
              'max_leaf_nodes':           [None], 
              'max_samples':              [None],
              'min_impurity_decrease':    [0.0], 
              'min_impurity_split':       [None],
              'min_samples_leaf':         [1,2], 
              'min_samples_split':        [2],
              'min_weight_fraction_leaf': [0.0], 
              'n_estimators':             [100, 200, 300, 500, 1000],
              'n_jobs':                   [None], 
              'oob_score':                [False], 
              'random_state':             [1994],
              'verbose':                  [0], 
              'warm_start':               [True, False]
              }
                                           
clf2 = GridSearchCV(rfc, parameters2)

# =============================================================================
# Plot ROC curves
# =============================================================================
from sklearn import svm, tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


clf_svm = svm.SVC()
clf_tre = tree.DecisionTreeClassifier(random_state = 1994) 
clf_mlp = MLPClassifier(hidden_layer_sizes = (8, 16, 32, 2), random_state=1994, max_iter=500, 
                        activation = 'tanh', 
                        solver = 'adam', 
                        learning_rate = 'adaptive')
clf_rfe = RandomForestClassifier(random_state=1994)


# set up a train-dev and a test-set
from sklearn.model_selection import train_test_split
X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=.2,
                                                    random_state=1994)
# find the best rf
clf2.fit(X_train_dev,y_train_dev)
clf_svm.fit(X_train_dev,y_train_dev)
clf_tre.fit(X_train_dev,y_train_dev)
clf_mlp.fit(X_train_dev,y_train_dev)
clf_rfe.fit(X_train_dev,y_train_dev)

# get predict propabilitie for each clf
y_pred_proba = clf2.predict_proba(X_test)
y_pred_svm = clf_svm.decision_function(X_test)
y_pred_tre = clf_tre.predict_proba(X_test)
y_pred_mlp = clf_mlp.predict_proba(X_test)
y_pred_rfe = clf_rfe.predict_proba(X_test)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# get roc curves
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,1])
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
fpr_tre, tpr_tre, _ = roc_curve(y_test, y_pred_tre[:,1])
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_pred_mlp[:,1])
fpr_rfe, tpr_rfe, _ = roc_curve(y_test, y_pred_rfe[:,1])

plt.style.use('ggplot')

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rfe, tpr_rfe, label='Random Forest Classifier')
plt.plot(fpr_svm, tpr_svm, label='Support Vector Machine')
plt.plot(fpr_tre, tpr_tre, label='Decision Tree')
plt.plot(fpr_mlp, tpr_mlp, label='Multilayer Perceptron')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

plt.savefig('results/plots/roc_curve.svg', format = 'svg')
plt.show()
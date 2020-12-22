# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:43:50 2020

@author: Johannes Allgaier

Julius Maximilian University of Würzburg
Am Schwarzenberg 15 / A15.4.311
97078 Würzburg
Germany

Phone | +49 931 201 46407
E-Mail | johannes.allgaier@uni-wuerzburg.de


Purpose of this file

    
"""

# activate venv using 'conda activate name_of_venv'

# import required packages

# =============================================================================
# Second approach for prediction
# =============================================================================
import pandas as pd
from sklearn import svm, tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import modules as m


# read in data
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')

# define feature_lists with [q1], [q1, q2],..., [q1,q2,..., q7]
feature_lists = []
for i in range(1,8):
    ls = []
    for j in range(i):
        ls.append(f'question{j+1}')
    feature_lists.append(ls)

ls = feature_lists[6]
for q in range(8):
    feature_lists.append(ls + [f'question8_{q}'])
    
# define clfs
clf_svm = svm.SVC()
clf_tre = tree.DecisionTreeClassifier() 
clf_mlp = MLPClassifier(hidden_layer_sizes = (8, 16, 32), random_state=1, max_iter=500, 
                        activation = 'tanh')
clf_rfe = RandomForestClassifier(random_state=1)

# pack them into a list
clfs = [clf_svm, clf_tre, clf_mlp, clf_rfe]
clfs_name = ['SVM', 'TREE', 'MLP', 'RANDOM FOREST']

# =============================================================================
# for each feature_list train and predict the clfs
# =============================================================================
pagebreak = f'###################################################################\n'

f = open('results_prediction_2.txt', 'w+')  
f.write('Classifiers:\n')
f.write(f'{clf_svm}\n\n{clf_tre}\n\n{clf_mlp}\n\n{clf_rfe}\n\n')
 
for feature_list in feature_lists:
    f.write(pagebreak)
    f.write(f'Features:\n{feature_list}\n')
    f.write(pagebreak)
    for c, clf in enumerate(clfs):

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
        
        # instantiate a classifier
        clf_name = clfs_name[c]
        
        # fit classifier to data
        clf.fit(X_train, y_train)
        
        # predict
        y_pred = clf.predict(X_test) 
        
        # evaluate
        from sklearn.metrics import confusion_matrix
        confusion = pd.DataFrame(data = confusion_matrix(y_test, y_pred),
                                 index = ['Male_true', 'Female_true'],
                                 columns = ['Male_pred', 'Female_pred'])
        
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, target_names = ['Male', 'Female'])
        
        # save results into file
        f.write(f'{clf_name}\n\n')
        f.write(f'confusion matrix:\n{confusion}\n\n')
        f.write(f'classification report:\n{report}\n\n')
f.close()




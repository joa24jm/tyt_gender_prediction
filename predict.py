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

# activate venv using 'conda activate venv' in the console

# import packages
import pandas as pd
from sklearn import svm, tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import modules as m
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt


data = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')

# to approaches: one with and one without 8_5
features = [
    data.columns.tolist()[2:8],
    data.columns.tolist()[1:8] + ['question8_5']
    ]

approaches = ['q2_q7', 'q1_q8_5']

f = open('20-07-10_results_prediction.txt', 'w+')  
f.write('Features:\n')
pagebreak = f'###################################################################\n'

for i, feature in enumerate(features):
    
    f.write(pagebreak)
    f.write(f'Features:\n{feature}\n')
    f.write(pagebreak)
    approach = approaches[i] 
    
    # import dataframe
    df = data[feature + ['gender']]
    
    # drop NULL values
    df = df.dropna(axis='index', how = 'any')
        
    
    # 50% male, 50% female in target for unbiased estimation
    df = m.equal_split(df)
    
    f.write(f'data shape: {df.shape}\n')
    f.write(pagebreak)
    
    # specify features and target
    X = df[feature]
    y = df['gender']
    
    # specify train-test set
    X_train, X_test, y_train, y_test = m.train_test_set(X,y)
    
    # instantiate the models
    clf_svm = svm.SVC()
    clf_tre = tree.DecisionTreeClassifier(random_state = 1994) 
    clf_mlp = MLPClassifier(hidden_layer_sizes = (8, 16, 32, 2), random_state=1994, max_iter=500, 
                            activation = 'tanh', 
                            solver = 'adam', 
                            learning_rate = 'adaptive')
    clf_rfe = RandomForestClassifier(random_state=1994)
    
    # pack them into a list
    clfs = [clf_svm, clf_tre, clf_mlp, clf_rfe]
    
    f.write(pagebreak)
    
    for clf in clfs:
        f.write(f'\n Setup \n')
        f.write(str(clf))
        f.write(pagebreak)
        
    
    # save results and set_ups into a dict
    keys = ['Support Vector Machine', 'Decision Tree', 'Multilayer Perceptron Regressor', 'Random Forest']
    
    # trained clfs
    trained_clfs = []
    
    for key, clf in zip(keys, clfs):

        
        # fit clf on train data
        clf.fit(X_train, y_train)
        
        # append trained clf
        trained_clfs.append(clf)
        
        # predict on test data
        y_pred = clf.predict(X_test)
        
        # create confusion matrix
        confusion = pd.DataFrame(data = confusion_matrix(y_test, y_pred),
         index = ['Male_true', 'Female_true'],
         columns = ['Male_pred', 'Female_pred'])
        
        # create a report
        report = classification_report(y_test, y_pred, target_names = ['Male', 'Female'])
        
        # write into txt
        f.write(f'\n{key}\n')
        f.write(f'confusion matrix:\n{confusion}\n\n')
        f.write(f'classification report:\n{report}\n\n')
        f.write(pagebreak)
        
f.close()
    
# print res to confusion matrix
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, 
                                             sharex = True, sharey = True)
# pack subfigs into a list
axis = [ax1, ax2, ax3, ax4]

for clf, ax, key in zip(trained_clfs, axis, keys):
    
    # plot confusion matrix
    disp = plot_confusion_matrix(clf, X_test, y_test, 
                                 display_labels = ['Male', 'Female'],
                                 cmap = plt.cm.Blues,
                                 normalize = 'true')
    disp.ax_.set_title(f'{key}')
    
    plt.savefig(f'results/plots/con_mat_{key}.svg', 
            format = 'svg')























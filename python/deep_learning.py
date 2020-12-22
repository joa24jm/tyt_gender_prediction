# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:23:48 2020

@author: Johannes Allgaier

Julius Maximilian University of Würzburg
Am Schwarzenberg 15 / A15.4.311
97078 Würzburg
Germany

Phone | +49 931 201 46407
E-Mail | johannes.allgaier@uni-wuerzburg.de


Purpose of this file

Use Keras for a binary classification of the TYT dataset

    
"""
# import required packages
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')

# define X and y
X_features = list(df.iloc[:,1:8].columns) + ['question8_5']
X = df[X_features]
y = df['gender']

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(X.shape[1], input_dim=X.shape[1], 
                 activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', 
               metrics=['accuracy'])
	return model


# evaluate model
estimator = KerasClassifier(build_fn=create_baseline, 
                            epochs=20, batch_size=1000, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, 
                                     results.std()*100))

# Baseline: 0.00% (0.00%)
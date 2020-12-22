# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:21:28 2020

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
import numpy as np

def replace_wrong_values(s):
    """

    Parameters
    ----------
    s : series like one column of a df, i.e. df['question']

    Returns
    -------
    df : dataframe with correct values.

    """
    # define values
    ranges = [0, .125, .25, .375, .5, .625, .75, .875, 1]
    
    # define borders 
    borders = [0, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 1]
    
    for i in range(len(borders[:-1])):
        # get all indices where the value is smaller than range value
        indices = s[(s < borders[i+1]) & (s >= borders[i])].index
        
        # replace this value with the correct one
        s.loc[indices] = ranges[i]
        
    return s

def missing_value_treatment(df, columns):
    """
    Try KNN to replace missing values 
    If there are more than three rows from one user with missing values, 
    try KNN within the user's data.
    If there are less than three rows from one user, try median and most_frequent for the whole 
    dataset.
    
    Parameters
    ----------
    df that contains features with missing values.

    Returns
    -------
    df with no missing values

    """
    # import KNN from sklearn
    from sklearn.impute import KNNImputer, SimpleImputer

    
    slider_questions = ['question2', 'question3', 'question4', 'question5',
                     'question6', 'question7']
    bool_questions = ['question1', 'question8']
    
    # set up imputation strategies
    imp_knn = KNNImputer(n_neighbors = 2, missing_values = np.nan)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    
    imp_mean.fit(df[slider_questions])
    imp_mf.fit(df[bool_questions])

    
    # get all users from this dataset
    user_ids = df['user_id'].unique().tolist()
    
    # for each user
    for user in user_ids:

        # select user
        filt = df['user_id'] == user
        X = df[filt][columns]

        # fit imputers
        imp_mean.fit(df[slider_questions])
        imp_mf.fit(df[bool_questions])
        
        # check for any nan values for this user
        if X.isnull().values.any():
            
            # check if this user has sufficient (>1 ) notnull values per column
            # 1: Get count for notnull values per column for this user
            not_null_per_column = [X[col].notnull().sum() for col in X.columns]
            
            X_sli = X[slider_questions]
            X_bol = X[bool_questions]
            
            # 2: if any of the columns contains 1 or less not null values, use whole df to calculate mean and mf   
            if False in [count > 1 for count in not_null_per_column]:
                df.loc[X_sli.index, X_sli.columns] = imp_mean.transform(X_sli)          
                df.loc[X_bol.index, X_bol.columns] = imp_mf.transform(X_bol)                
                
            else : # X.shape[0] > 2 (if user data is sufficient, use user-specific KNN) 
                df.loc[X_sli.index, X_sli.columns] = imp_knn.fit_transform(X[slider_questions])
                df.loc[X_bol.index, X_bol.columns] = imp_mf.fit_transform(X[bool_questions])
                
    return df

def create_features(df):
    """

    Parameters
    ----------
    df : dataframe that contains user_id, q1, q2, ..., q7

    Returns
    dataframe with cols
        user_id, q1_mean, q1_std, q2_mean, q2_std, ..., q7_mean, q7_std
    -------
    Try to summererize each information per user in orer to get better prediction results

    """
    # import
    import pandas as pd
    
    # replace boolean values
    df.question1.replace({True:1, False:0}, inplace = True)
    
    # get features
    feature_list = ['question1', 'question2', 'question3', 'question4', 
                    'question5', 'question6', 'question7']
    
    # create a matrix that contains the features
    info = df[feature_list].describe()
    features = ['user_id']
    
    for col in info.columns.tolist():
        for index in info.index.tolist():
            features.append(f'{col}_{index}')   
    
    #
    df_return = pd.DataFrame(data = None, columns = features )
    
    # get all users from this dataset
    user_ids = df['user_id'].unique().tolist()
    
    # for each user
    for user in user_ids:
        # get sub df with only one user
        X = df[df['user_id'] == user]
        user_info = X[feature_list].describe()
        to_append = [user]
        for col in user_info.columns.tolist():
            for index in user_info.index.tolist():
                to_append.append(user_info.loc[index, col])
        s = pd.Series(to_append, index = features)
        df_return = df_return.append(s, ignore_index = True)
    
    df_return.rename(columns = {'question1_count':'n_filled_questionaires'}, 
                     inplace = True)
    df_return.drop(['question2_count', 'question3_count', 'question4_count',
                    'question5_count', 'question6_count', 'question7_count'],
                   axis = 1, inplace = True)
        
    return df_return
                
def equal_split(df):
    """
    Split up df into 50% male and 50% female users
    Parameters
    ----------
    df : preprocessed df.

    Returns
    -------
    df with equal man and women.

    """
        
    # sample randomly 20,000 male and take all 20,000 female users 
    df_f = df[df['gender'] == 1].sample(frac=1, random_state = 12345) # all female users
    df_m = df[df['gender'] == 0].sample(n=df_f.shape[0], random_state = 12345) # n = n_female_users

    df = df_m.append(df_f)

    return df

def train_test_set(X, y, seed):
    """
    Split up X and y into a train and development set
    """
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    y = np.reshape(y.values, -1) # reshape y to (n_samples, )

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.20, 
                                                        random_state=seed,
                                                        shuffle = True)
    return X_train, X_test, y_train, y_test


def train_clf(clf, X_train, y_train):
    """
    trains a sklearn classifier given X_train and y_train data

    Parameters
    ----------
    clf : sk learn classifier
    X_train : train data as received from train_test_split
    y_train : train data as received from train_test_split.

    Returns
    -------
    trained clf.

    """    
    return clf.fit(X_train, y_train)

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    from itertools import chain, combinations
    s = list(iterable)
    chn = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    chn = [list(el) for el in chn]
    del chn[0]
    return chn

def make_all_possible_feature_combinations(ls1, ls2):
    """
    ls1 = powerset(q1, q2, ..., q7)
    ls2 = [q8_0, q8_1, ..., q8_7]
    """
    chn = powerset(ls1)
    res = chn.copy()
    
    for el in chn:
        for el2 in ls2:
            res.append(el + [el2])
            
    return sorted(res)



        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
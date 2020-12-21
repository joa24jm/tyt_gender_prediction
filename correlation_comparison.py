# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:42:33 2020

@author: Johannes Allgaier

Find the most important feature to predict the gender using
- correlation (linregress and pointserial)
- accuracy
- permutation importance

"""
#%%
# import modules
import pandas as pd
import modules as m
import scipy.stats as ss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

#%%
def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))



def calc_corr(x, y, x_datatype, y_datatype, x_label, y_label):
    """
    Calculate the correlation of two arrays x and y

    Parameters
    ----------
    x : array with vals
    y : array with vals
    x_datatype: binary, discrete or continous
    y_datatype: binary, discrete or continous
    
    Returns
    -------
    dict with pval, rval and method

    """
    
    from scipy import stats
    
    rval = None
    pval = None
    method = None

    
    if (y_datatype in ['binary', 'discrete'] and x_datatype == 'continous') or (y_datatype == 'continous' and x_datatype in ['binary', 'discrete']):
        
        # Calculate a point biserial correlation coefficient
        rval = stats.pointbiserialr(x, y)[0]
        pval = stats.pointbiserialr(x, y)[1]
        method = 'pointbiserial'
        
        # format for scientific notation
        pval = "{:.2e}".format(pval)
    
    if (y_datatype in  ['binary', 'discrete'] and x_datatype in ['binary', 'discrete']):
        # Calculate pearson correlation
        rval = cramers_v(pd.crosstab(x,y).to_numpy())
        pval = 'None' # not yet implemented
        method = 'cramers_v'
        
    if y_datatype == 'continous' and x_datatype == 'continous':
        rval = stats.pearsonr(x,y)[0]
        pval = stats.pearsonr(x,y)[1]
        method = 'pearson'
    

    
    return {'rval':rval, 'pval':pval, 'method':method}

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
    
    result_dic = {'importance': result.importances_mean[0],
                  'std':        result.importances_std[0]}
    
    return result_dic
#%%
def main():
    
    global results

    # import dataset
    df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')
    
    # import and prepare codebook
    cb = pd.read_excel('codebook/codebook_johannes.xlsx', sheet_name='codebook')
    cb = cb[['question_id', 'question / meaning', 'datatype']].iloc[0:17]
    cb.rename(columns={'question_id':'label', 
                       'question / meaning': 'meaning'}, inplace = True)
    
    cb_dat = dict(zip(cb.label, cb.datatype))
    cb_dat['gender'] = 'binary'
    
    
    # get all features of df
    features = df.iloc[:,1:-1].columns.tolist()
    
    # drop q8_3 as this is the FUP question to 'none of these worst symptoms'
    features.remove('question8_3')
    

    # get stats about the df
    filt_male = df['gender'] == 0
    filt_female = df['gender'] == 1
    
    mean_male   = df[filt_male].mean()
    mean_female = df[filt_female].mean()
    std_male    = df[filt_male].std()
    std_female  = df[filt_female].std()
    
    stats = pd.concat([mean_male, mean_female, std_male, std_female], axis = 'columns').rename(columns = {
        0: 'mean_male', 1 : 'mean_female', 2: 'std_male', 3: 'std_female'})
    
    # calculate t.test and pvalue for stats df
    for idx in stats.index:
        # df[filt_male][col].values
        # to-do: get two lists of male and female vals per feature and calculate the t.test and pval
        # i.e. - is there a difference between the means?
        
        # number of answers per gender
        num_1 = len(df[filt_female][idx].dropna())
        num_0 = len(df[filt_male][idx].dropna())
        num_min = min(num_1, num_0)
        
        # get the same number of answers per sample
        sample_1 = df[filt_female][idx].dropna().sample(n=num_min)
        sample_0 = df[filt_male][idx].dropna().sample(n=num_min)
        
        # calculate the ttest| h0: both samples have the same mean
        tval, pval = ss.ttest_ind(sample_1, sample_0)
        
        # calculate cohen's alias effect size
        mean_1 = np.mean(sample_1)
        mean_0 = np.mean(sample_0)
        std = ((np.var(sample_0) + np.var(sample_1))/2)**0.5
        
        if idx == 'gender':
            continue
        
        effect_size = (mean_0-mean_1)/std
        
        stats.loc[idx, 'tval'] = f't({num_min-1}) = {tval})'
        stats.loc[idx, 'pval'] = '<.001' if pval < 0.001 else round(pval, 3)
        stats.loc[idx, 'effect_size'] = effect_size
        
    # drop vals for gender
    stats.loc['gender', ['tval', 'pval']] = 'nan'
    
    # define results dict
    results = pd.DataFrame(index = features, columns = ['permutation_importance', 'permutation_std', 
                                                        'forest_accuracy', 'forest_std',
                                                        'correlation_r-value',
                                                        'correlation_p-value',
                                                        'correlation_method'])
    
    
    
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
        clf = RandomForestClassifier(random_state  = seed)
        fitted_clf = clf.fit(X_train, y_train)
        
        # calculate permutation importance
        result_dic = calculate_permutation_importance(fitted_clf, X_test, y_test)
        
        # calculate random forest feature importance
        accuracy_means = cross_val_score(clf, X, y, scoring = 'accuracy', cv = 5)
        accuracy_mean  = np.mean(accuracy_means)
        
        # calculate correlation
        correlation_dic = calc_corr(X.reshape(-1), y, cb_dat[col], cb_dat['gender'], col, 'gender')
        
        # append score to results list
        results.loc[col, 'permutation_importance'] = result_dic['importance']
        results.loc[col, 'permutation_std'] =        result_dic['std']
        results.loc[col, 'forest_accuracy'] =        accuracy_mean
        results.loc[col, 'forest_std']      =        np.std(accuracy_means)
        results.loc[col, 'correlation_r-value'] =    correlation_dic['rval']
        results.loc[col, 'correlation_p-value'] =    correlation_dic['pval']
        results.loc[col, 'correlation_method'] =     correlation_dic['method']
        
        # rank the variable importanc
        results['permutation_rank'] = results['permutation_importance'].rank(ascending=False)
        results['forest_rank'] = results['forest_accuracy'].rank(ascending = False)
        results['correlation_rank'] = results['correlation_r-value'].abs().rank(ascending = False)
        
    # concatenate df objects stats and results
    results = pd.concat([results, stats], axis = 'columns').drop(index = ['user_id', 'question8_3', 'gender'])
    
    # save results
    results.to_excel('results\\feature_importance_comparison.xlsx')


# only run functions on demand
if __name__ == "__main__":
    main()
    
    






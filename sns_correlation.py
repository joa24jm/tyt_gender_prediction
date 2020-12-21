# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:33:50 2020

@author: Johannes Allgaier
"""


# import 
import pandas as pd
from correlation_comparison import calc_corr

# main function
# def main():
    
# import dataset
global df
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')

# drop cols
df.drop(columns=['user_id', 'question8_3'], inplace = True)

# import and prepare codebook
global cb
cb = pd.read_excel('codebook/codebook_johannes.xlsx', sheet_name='codebook')
cb = cb[['question_id', 'question / meaning', 'datatype']].iloc[0:17]
cb.rename(columns={'question_id':'label', 
                   'question / meaning': 'meaning'}, inplace = True)

# create a codebook dict for easier access
global cb_dic
cb_dic = dict(zip(cb.label, cb.meaning))

global cb_dat
cb_dat = dict(zip(cb.label, cb.datatype))
cb_dat['gender'] = 'binary'

#%%


def create_corr_matrix(df, cb_dat):
    
    # create a list of cols
    ls_of_cols = list(df.columns)
    
    corr_mat = pd.DataFrame(data = None, index = ls_of_cols, columns = ls_of_cols)

    for col1 in ls_of_cols:
        for col2 in ls_of_cols:
            
            rval = None
            
            if col1 == col2:
                rval = 1
            
            else: # col1 != col2
                
                # get valid values for cols
                vals1 = df[[col1, col2]].dropna()[col1].values
                vals2 = df[[col1, col2]].dropna()[col2].values
    
                # get datatype for cols
                dtype1 = cb_dat[col1]
                dtype2 = cb_dat[col2]
                
                if len(vals1) > 1:                            
                    rval = calc_corr(vals1, vals2, dtype1, dtype2, col1, col2)['rval']
                
            corr_mat.loc[col1, col2] = rval
            
    return corr_mat
    
#%%        
corr_mat = create_corr_matrix(df, cb_dat)

# format solution
corr_mat = corr_mat.astype('float').round(3)

corr_mat.to_excel('results\df_correlation_values.xlsx')



# # run script
# if __name__ == "__main__":
#     main()
    

    
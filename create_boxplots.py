# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:32:42 2020

@author: Johannes Allgaier
"""
import pandas as pd
import matplotlib.pyplot as plt


# import dataset
df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx')

# remove question8_3 
df.drop(columns = 'question8_3', inplace = True)

df.gender.replace({0: 'male', 
                   1: 'female'}, inplace = True)

# define feature cols
cols = df.columns[1:-1]

# create a grid with 15 boxplots grouped by gender
boxplot = df.boxplot(column = list(cols),
                     rot = 90, 
                     by = 'gender', 
                     fontsize = 'small')

plt.subplots_adjust(hspace = 0.4)

plt.style.use('default')
plt.style.use('seaborn-dark-palette')

plt.show()

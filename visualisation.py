# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:54:37 2020

@author: Johannes Allgaier

Julius Maximilian University of Würzburg
Am Schwarzenberg 15 / A15.4.311
97078 Würzburg
Germany

Phone | +49 931 201 46407
E-Mail | johannes.allgaier@uni-wuerzburg.de


Purpose of this file:
    
Visualize insights of tinnitus dataset

    
"""
"""
# =============================================================================
# Plot worst symptoms from q8 grouped by gender
# =============================================================================
import pandas as pd

ans = pd.read_excel('codebook/codebook_johannes.xlsx',
                    sheet_name = 'answers',
                    na_values = [-0.01, 'nan'])

# select question 24: 'Please tick only the worst symptom'
filt = ans['question_id'] == 24
q24 = ans[filt]

# find all possible answers except the last one (last on is a bug answer that contains all answers)
answers = q24['answer'].unique().tolist()[:-1]

# shorten answers for better visualisation
shortened = [
    'Hard for me to get to sleep',
    'Feeling depressed',
    'Harder to relax',
    'None of these symptoms',
    'Strong worries',
    'Difficult to follow a conversation, a piece of music or a film',
    'Difficult to concentrate',
    'I am more irritable with my family, friends and colleagues',
    'I am more sensitive to environmental noises']

q5 = ans[['user_id', 'answer']][ans['question_id'] == 5] # Gender

q5_q24 = q5.merge(q24[['user_id', 'answer']], how = 'left', on = 'user_id')

# only keep users that have an id
filt = q5_q24['user_id'].isnull() == False
q5_q24 = q5_q24[filt]

# groupby answer and gender
male_group = q5_q24[['answer_x', 'answer_y']][q5_q24['answer_x'] == 'Male'].groupby('answer_y').agg('count')
female_group = q5_q24[['answer_x', 'answer_y']][q5_q24['answer_x'] == 'Female'].groupby('answer_y').agg('count')

# chose only index from 'answers'
female_group = female_group.loc[answers]
male_group = male_group.loc[answers]

# create list of vals to plot
female_vals = female_group['answer_x'].values.tolist()
male_vals = male_group['answer_x'].values.tolist()

# in percent
res_f, res_m = [], []
for f, m in zip(female_vals, male_vals):
    res_f.append(round(f/sum(female_vals),5))
    res_m.append(round(m/sum(male_vals),5))

# =============================================================================
# plot results
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

fig, ax = plt.subplots(figsize = (8,4))

y_pos = np.arange(len(answers))
width = 0.3

ax.barh(y_pos, res_f, width, label = 'Female',
        color = '#ff8d33', align = 'center')
ax.barh(y_pos + width, res_m, width, label = 'Male', 
        color = '#29713f', align = 'center')
ax.set_yticks(y_pos)
ax.set_yticklabels(shortened)
ax.invert_yaxis() # labels read top to bottom
ax.set_xlabel('Percent')
ax.set_title('Worst symptom grouped by gender')
ax.legend()

plt.tight_layout()

plt.savefig('results/plots/worst_symptom_grouped_by_gender_percent.svg', 
            format = 'svg')

plt.show()

# second approach

category_names = shortened

results = {'male': res_m, 'female': res_f}

def plot_results(results, category_names):
    
    # Parameters
    # ----------
    # results : dict
    #     Mapping from gender to relative responses per answeroption
    # category_names : list
    #     shortened answer options

    # Returns
    # -------
    # visualisation as horizontal barh plot

    
    plt.style.use('seaborn')
    
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('twilight')(
        np.linspace(.15, .85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))

    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height = .8,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, '{:.1%}'.format(c), ha='center', va='center',
                    color=text_color)
            
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.18, 
                     box.width, box.height * 0.9])
    
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.32), 
              fontsize='medium', fancybox = True)
    
    plt.title('Worst symptom by gender')

    return fig, ax

plot_results(results, category_names)

plt.savefig('results/plots/worst_symptom_grouped_by_gender_stacked.svg', 
            format = 'svg')
plt.show()


# =============================================================================
# Plot other results
# =============================================================================

df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx',
                    sheet_name = 'Sheet1')

import matplotlib.pyplot as plt
import numpy as np

df['question1'].replace(to_replace={0:'No', 1:'Yes'}, inplace = True)
df['gender'].replace(to_replace={0:'Male', 1:'Female'}, inplace = True)

filt_m = df['gender'] == 'Male'
filt_f = df['gender'] == 'Female'

q1_m = df[filt_m][['gender', 'question1']].groupby('question1').agg('count')
q1_f =  df[filt_f][['gender', 'question1']].groupby('question1').agg('count')

# calculate ratios
male_yes = q1_m.iloc[1,0]/q1_m.sum()[0]
male_no =  q1_m.iloc[0,0]/q1_m.sum()[0]

female_yes = q1_f.iloc[1,0]/q1_f.sum()[0]
female_no =  q1_f.iloc[0,0]/q1_f.sum()[0]

ind = np.arange(2)
width = .5

plt.style.use('default')
plt.rcParams.update({'font.size': 10})

p1 = plt.bar(ind, [male_yes, female_yes], width)
p2 = plt.bar(ind, [male_no, female_no], width,
             bottom=[male_yes, female_yes])

plt.ylabel('Ratio')
plt.title('Did you perceive the tinnitus right now?')
plt.xticks(ind, ('Male', 'Female'))
plt.yticks(np.arange(0, 1.1, .1))

plt.legend((p1[0], p2[0]), ('Yes', 'No'))

plt.savefig('results/plots/q1_grouped_by_gender.svg', 
            format = 'svg')

plt.show()


# =============================================================================
# Histograms for q2, q3, ..., q7
# =============================================================================

df['question1'].replace(to_replace={0:'No', 1:'Yes'}, inplace = True)
df['gender'].replace(to_replace={0:'Male', 1:'Female'}, inplace = True)

filt_m = df['gender'] == 'Male'
filt_f = df['gender'] == 'Female'

bins = [round(el, 2) for el in list(np.arange(0, 1.1, .1))]

# calculate how many users answer in a range of stepsize .05 per gender
q2_m = df[filt_m]['question2'].value_counts(bins = bins, normalize = True).values
q2_f = df[filt_f]['question2'].value_counts(bins = bins, normalize = True).values

# X pos
X_ = np.arange(len(bins))


# calulate relative answers
data = np.array([q2_m, q2_f])

plt.bar(bins[:-1], data[0].tolist(), edgecolor='black', label = 'Male', width = .025)

plt.xlabel('0 = minimu loudness, 1 = maximum loudness')
plt.ylabel('Percent')
plt.title('How loud is the tinnitus right now?')
plt.legend()
plt.xticks(ticks = bins, labels = bins, rotation = 90)
plt.xlim(0, 1)
plt.grid(False)
plt.show()
"""
# =============================================================================
# SNS heatmap
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

df = pd.read_excel('data/df_q1_q8_0_q8_8.xlsx',
                    sheet_name = 'Sheet1')

df.drop(columns = ['user_id', 'question8_3'], inplace = True)

# import and prepare codebook
cb = pd.read_excel('codebook/codebook_johannes.xlsx', sheet_name='codebook')
cb = cb[['question_id', 'question / meaning', 'datatype']].iloc[0:17]
cb.rename(columns={'question_id':'label', 
                   'question / meaning': 'meaning'}, inplace = True)

# create a codebook dict for easier access
cb_dic = dict(zip(cb.label, cb.meaning))

cb_dat = dict(zip(cb.label, cb.datatype))
cb_dat['gender'] = 'binary'

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
    
    if (y_datatype == 'binary' and x_datatype == 'continous') or (y_datatype == 'continous' and x_datatype == 'binary'):
        
        # Calculate a point biserial correlation coefficient
        rval = stats.pointbiserialr(x, y)[0]
        pval = stats.pointbiserialr(x, y)[1]
        method = 'pointbiserial'
        
        # print(f'pointbiserialr for {x_label} and {y_label}')
    
    else:
        # Calculate pearson correlation
        rval = stats.linregress(x, y)[2]
        pval = stats.linregress(x, y)[3]
        method = 'linregress'
    
    # format for scientific notation
    pval = "{:.2e}".format(pval)
    
    return {'rval':rval, 'pval':pval, 'method':method}

# create an empty corr df
corr_df = pd.DataFrame(columns = df.columns, index = df.columns)

for row in corr_df.index:
    for col in corr_df.columns:
       
        if row == col:
           corr = 1
       
        else:
           
           # get the vals with both series not na
           x = df[[row, col]].dropna(how='any')[row].values.tolist()
           y = df[[row, col]].dropna(how='any')[col].values.tolist()
           
           # check if intersection of indices < 2
           if len(x) < 2:
               continue
           
           # get datatypes from dic
           x_datatype = cb_dat[row]
           y_datatype = cb_dat[col]
           
           # calculate correlation
           corr = calc_corr(x, y, x_datatype, y_datatype, row, col)
            
        # safe val to corr_df
        corr_df.loc[row, col] = corr
        
corr_df.to_excel('results\corr_df.xlsx')  
        
        
# create seaborn heatmap to see correlations
corrMatrix = corr_df




# # Generate a mask for the upper triangle
# mask = np.zeros_like(corrMatrix, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True

# sns.set(style = 'white',
#         context = 'paper', 
#         color_codes = True)

# ax = plt.subplot()
# ax = sns.heatmap(round(corrMatrix, 2),
#                   annot = False,
#                   annot_kws={"size": 8, 
#                              'color':'black'},
#                   cmap = plt.cm.RdYlGn, 
#                   linecolor = 'lightgrey',
#                   linewidths = .01, 
#                   square = False, 
#                   mask = mask)


# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corrMatrix, dtype=bool))

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corrMatrix, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})




# k = 0
# for i in range(len(corr)):
#     for j in range(i+1, len(corr)):
#         k=k+1
#         s = "{:.2f}".format(corr.values[i,j])
#         ax.text(j+0.5,i+0.5,s, 
#             ha="center", va="center", size = 8)


# ax.set_title('Pearson resp. Pointbiserial correlation matrix')

# plt.tight_layout()

# plt.savefig('results/plots/sns_heatmap.svg', 
#             format = 'svg')

# plt.show()




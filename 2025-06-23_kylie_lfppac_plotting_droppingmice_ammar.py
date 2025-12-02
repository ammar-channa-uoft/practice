#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:49:00 2024

@author: mbellvila
"""

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import scipy.stats

pval = 0.05



#already ran data
directories_old = sorted(glob.glob('/home/achanna/projects/rrg-bojana/bellvila/Kylie/202*/'))
directories_old = directories_old[:-4] #drop last 4

#newly run data
directories_new = [path.replace('_pwrsp_dec.pickle', '_') for path in sorted(glob.glob('/scratch/achanna/kylie_edits/2025-*pwrsp_dec.pickle'))]

#combine
directories = directories_old + directories_new

#list of mice/dates to exclude from analysis (date, stimulus condition)
micetoexclude = [['2024-08-22', 'C'],
                 ['2024-08-22', 'S'],
                 ['2024-08-01', 'C'],
                 ['2024-07-26', 'C']]

#convert exclusion list above into a table
micetoexclude = pd.DataFrame(micetoexclude, columns = ['date','stim'])

lows = ['δ', 'θ', 'α', 'β']    			#list of low frequency PAC bands
highs = ['low-γ', 'high-γ']    			#list of high frequency PAC bands

lowlows = ['delta', 'theta', 'alpha', 'beta']   #labels for low frequency PAC bands
highhighs = ['lowgamma', 'highgamma']		#labels for high frequency PAC bands

powspecfrqs = [0.5, 4, 8, 12, 27, 80, 140]		#frequencies (in Hz) used for power spectral density calculations
powspecbands = ['δ', 'θ', 'α', 'β', 'low-γ', 'high-γ']	#labels for the above frequencies
lowfrqs = [4, 8, 12, 27]				#low frequency centers for PAC calculation
highfrq = 80						#high frequency cutoff for PAC calculation

#colour scheme for each stimulus condition's plot
colourscontrol = ['#1C1763', '#1B52BE', '#4FB1D4', '#8ae6d9']
coloursstim = ['#c21817','#ff7417','#ff9c00', '#fcd853']

#helper function to recolor matplotlib boxplot components
def bpcol(box_plot, edge_color):
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(box_plot[element], color=edge_color)


# %%
mouseinfo = [['2024-02-01', '134503', 'ipsi', 'C', 1, '5', 'M'], 
             ['2024-02-01', '144906', 'contra', 'C', 1, '5', 'M'],
             ['2024-02-01', '163650', 'ipsi', 'S', 1, '5', 'M'],
             ['2024-02-01', '172311', 'contra', 'S', 1, '5', 'M'],
             ['2024-02-02', '154340', 'ipsi', 'C', 2, '6', 'F'],
             ['2024-02-02', '165531', 'contra', 'C', 2, '6', 'F'],
             ['2024-02-02', '172716', 'contra', 'C', 2, '6', 'F'],
             ['2024-02-08', '123210', 'contra', 'C', 8, '12', 'F'],
             ['2024-02-08', '132225', 'ipsi', 'C', 8, '12', 'F'],
             ['2024-02-08', '134504', 'ipsi', 'C', 8, '12', 'F'],
             ['2024-02-08', '150820', 'contra', 'S', 8, '12', 'F'],
             ['2024-02-08', '155030', 'ipsi', 'S', 8, '12', 'F'], 
             ['2024-07-10', '125536', 'ipsi', 'C', 1, '5', 'M'], 
             ['2024-07-10', '135322', 'contra', 'C', 1, '5', 'M'],
             ['2024-07-10', '153448', 'ipsi', 'S', 1, '5', 'M'],
             ['2024-07-10', '161853', 'contra', 'S', 1, '5', 'M'],
             ['2024-07-10', '163831', 'contra', 'S', 1, '5', 'M'], 
             ['2024-07-11', '120919', 'ipsi', 'S', 2, '6', 'F'],
             ['2024-07-11', '125033', 'contra', 'S', 2, '6', 'F'],
             ['2024-07-11', '143539', 'ipsi', 'C', 2, '6', 'F'],
             ['2024-07-11', '151925', 'contra', 'C', 2, '6', 'F'],
             ['2024-07-17', '115534', 'ipsi', 'S', 8, '12', 'F'], 
             ['2024-07-17', '124355', 'contra', 'S', 8, '12', 'F'],
             ['2024-07-17', '151330', 'ipsi', 'C', 8, '12', 'M'],
             ['2024-07-17', '155141', 'contra', 'C', 8, '12', 'M'],
             ['2024-07-25', '123722', 'ipsi', 'S', 1, '5', 'M'], 
             ['2024-07-25', '142729', 'contra', 'S', 1, '5', 'M'],
             ['2024-07-25', '161210', 'ipsi', 'C', 1, '5', 'M'], 
             ['2024-07-25', '165436', 'contra', 'C', 1, '5', 'M'],
             ['2024-07-26', '121412', 'ipsi', 'C', 2, '6', 'F'], 
             ['2024-07-26', '130432', 'contra', 'C', 2, '6', 'F'], 
             ['2024-07-26', '155829', 'ipsi', 'S', 2, '6', 'F'],
             ['2024-07-26', '164604', 'contra', 'S', 2, '6', 'F'],
             ['2024-08-01', '130511', 'ipsi', 'S', 8, '12', 'M'],
             ['2024-08-01', '134613', 'contra', 'S', 8, '12', 'M'],
             ['2024-08-01', '153352', 'ipsi', 'C', 8, '12', 'M'],
             ['2024-08-01', '161344', 'contra', 'C', 8, '12', 'M'],
             ['2024-08-09', '152116', 'ipsi', 'S', 31, '35', 'M'],
             ['2024-08-09', '160028', 'ipsi', 'S', 31, '35', 'M'],
             ['2024-08-09', '163525', 'contra', 'S', 31, '35', 'M'],
             ['2024-08-09', '183327', 'contra', 'C', 31, '35', 'F'],
             ['2024-08-09', '191006', 'ipsi', 'C', 31, '35', 'F'],
             ['2024-08-22', '125624', 'ipsi', 'C', 29, '34', 'F'],
             ['2024-08-22', '142337', 'contra', 'C', 29, '34', 'F'],
             ['2024-08-22', '163501', 'ipsi', 'S', 29, '34', 'M'],
             ['2024-08-22', '174320', 'contra', 'S', 29, '34', 'M'],
             ['2025-10-08', '123509', 'contra', 'C', 2, '6', 'M'], 	#2025-10-08 (2nd mouse was only stimulated on contra side)
	     ['2025-10-08', '133902', 'ipsi', 'C', 2, '6', 'M'],
	     ['2025-10-08', '160356', 'contra', 'S', 2, '6', 'F'],
	     ['2025-10-10', '135528', 'ipsi', 'C', 8, '12', 'M'],	#2025-10-10
             ['2025-10-10', '145410', 'contra', 'C', 8, '12', 'M'],
             ['2025-11-04', '125527', 'contra', 'S', 33, '37', 'M'],	#2025-11-04
	     ['2025-11-04', '134827', 'ipsi', 'S', 33, '37', 'M'],
	     ['2025-11-04', '155813', 'contra', 'C', 33, '37', 'M'],
	     ['2025-11-04', '164635', 'ipsi', 'C', 33, '37', 'M'],
             ['2025-11-05', '131628', 'contra', 'C', 34, '38', 'M'],	#2025-11-05
	     ['2025-11-05', '141358', 'ipsi', 'C', 34, '38', 'M'],
 	     ['2025-11-05', '162926', 'ipsi', 'S', 34, '38', 'M'],
	     ['2025-11-05', '172202', 'contra', 'S', 34, '38', 'M']
             ]

mouseinfodf = pd.DataFrame(mouseinfo, columns= ['date', 'subtime', 'hemisphere', 'stim', 'dayspostarrival','dpstim', 'sex'])

megapwrspdf = pd.DataFrame()
megapacdf = pd.DataFrame()




# %%


for directory in directories:
    pwrspvals = np.load(directory + 'pwrsp_dec', allow_pickle = True)
    pacvals = np.load(directory + 'pac_dec', allow_pickle = True)
    pwrspdf = pd.DataFrame(pwrspvals, columns = ['date', 'subtime', 'electrode', 'totdur', 'tp', 'pwrspwin', 'Fs', 'freq', 'pwrsp','resampled'])                                           
    pacdf = pd.DataFrame(pacvals, columns = ['date', 'subtime', 'electrode', 'tortdur', 'tp', 'pacwinsize', 'pacwintp', 'Fs', 'pacmethod', 'resampled', 
                                             'p', 'xpac', 'pval', 'xpac_smean'])
    for i in range(len(pwrspdf)):
        mousetouse = mouseinfodf[(mouseinfodf['date'] == pwrspdf.loc[i]['date'])&(mouseinfodf['subtime'] == pwrspdf.loc[i]['subtime'])]
        pwrspdf.loc[i,'hemi'] = mousetouse['hemisphere'].item()
        pwrspdf.loc[i,'stim'] = mousetouse['stim'].item()
        pwrspdf.loc[i,'dpa'] = mousetouse['dayspostarrival'].item() 
        pwrspdf.loc[i,'dps'] = mousetouse['dpstim'].item() 
        pwrspdf.loc[i, 'sex'] =  mousetouse['sex'].item() 


    for i in range(len(pacdf)):
        mousetouse = mouseinfodf[(mouseinfodf['date'] == pacdf.loc[i]['date'])&(mouseinfodf['subtime'] == pacdf.loc[i]['subtime'])]
        pacdf.loc[i,'hemi'] = mousetouse['hemisphere'].item()
        pacdf.loc[i,'stim'] = mousetouse['stim'].item()
        pacdf.loc[i,'dpa'] = mousetouse['dayspostarrival'].item()
        pacdf.loc[i, 'dps'] = mousetouse['dpstim'].item() 
        pacdf.loc[i, 'sex'] =  mousetouse['sex'].item() 
    megapwrspdf = pd.concat((megapwrspdf, pwrspdf), ignore_index = True)
    megapacdf = pd.concat((megapacdf, pacdf), ignore_index = True)

megapwrspdf.to_pickle("/scratch/achanna/kylie_edits/megapwspdf_ammarrun.pk1")
megapacdf.to_pickle("/scratch/achanna/kylie_edits/megapacdf_ammarrun.pk1")



#Dropping bad recordings
indexbad = megapwrspdf[(megapwrspdf['date'] == '2024-07-10') & (megapwrspdf['subtime'] == '163831')].index
megapwrspdf.drop(indexbad, inplace=True)
megapwrspdf = megapwrspdf.reset_index(drop = True)
indexbad = megapacdf[(megapacdf['date'] == '2024-07-10') & (megapacdf['subtime'] == '163831')].index
megapacdf.drop(indexbad, inplace=True)
megapacdf = megapacdf.reset_index(drop = True)

indexbad = megapwrspdf[(megapwrspdf['date'] == '2024-08-09') & (megapwrspdf['subtime'] == '152116')].index
megapwrspdf.drop(indexbad, inplace=True)
megapwrspdf = megapwrspdf.reset_index(drop = True)
indexbad = megapacdf[(megapacdf['date'] == '2024-08-09') & (megapacdf['subtime'] == '152116')].index
megapacdf.drop(indexbad, inplace=True)
megapacdf = megapacdf.reset_index(drop = True)

#dropping if outside 20minutes
indexAge = megapwrspdf[megapwrspdf['tp']*2 >= 20].index
megapwrspdf.drop(indexAge , inplace=True)
megapwrspdf = megapwrspdf.reset_index(drop = True)
indexAge = megapacdf[megapacdf['pacwintp'] > 10].index
megapacdf.drop(indexAge , inplace=True)
megapacdf = megapacdf.reset_index(drop = True)



#Channels 0 and 15 are empty on the pin
indexelectrode = megapwrspdf[megapwrspdf['electrode'] > 14].index
megapwrspdf.drop(indexelectrode , inplace=True)
megapwrspdf = megapwrspdf.reset_index(drop = True)
indexelectrode = megapacdf[megapacdf['electrode'] > 14].index
megapacdf.drop(indexelectrode , inplace=True)
megapacdf = megapacdf.reset_index(drop = True)

indexelectrode = megapwrspdf[megapwrspdf['electrode'] < 1].index
megapwrspdf.drop(indexelectrode , inplace=True)
megapwrspdf = megapwrspdf.reset_index(drop = True)
indexelectrode = megapacdf[megapacdf['electrode'] < 1].index
megapacdf.drop(indexelectrode , inplace=True)
megapacdf = megapacdf.reset_index(drop = True)


#%%


# Perform a left merge with indicator to identify rows to drop
merged = megapwrspdf.merge(micetoexclude, on=['date', 'stim'], how='left', indicator=True)
# Keep only rows that were not in the todrop dataframe
megapwrspdf = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
megapwrspdf = megapwrspdf.reset_index(drop = True)

merged =  megapacdf.merge(micetoexclude, on=['date', 'stim'], how='left', indicator=True)
megapacdf = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
megapacdf = megapacdf.reset_index(drop = True)


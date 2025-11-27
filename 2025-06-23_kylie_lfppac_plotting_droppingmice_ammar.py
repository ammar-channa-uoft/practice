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


##['2025-11-05', '131628', 'contra', 'C', 35, '39', 'M'],
['2025-11-05', '141358', 'ipsi', 'C', 35, '39', 'M'],
['2025-11-05', '162926', 'ipsi', 'S', 35, '39', 'M'],
['2025-11-05', '172202', 'contra', 'S', 35, '39', 'M']


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


#%%
def plotting_pos(dpstim):
    if float(dpstim) == 5:
        return 0
    elif float(dpstim) == 6:
        return 2
    elif float(dpstim) == 12:
        return 4
    elif float(dpstim) > 12:
        return 6
    
def plotting_label(dpstim):
    if float(dpstim) == 5:
        return '5'
    elif float(dpstim) == 6:
        return '6'
    elif float(dpstim) == 12:
        return '12'
    elif float(dpstim) > 12:
        return '35'

# Apply the function to the Age column using the apply() function
megapacdf['plotpos'] = megapacdf['dps'].apply(plotting_pos)
megapwrspdf['plotpos'] = megapwrspdf['dps'].apply(plotting_pos)
megapacdf['plotlabel'] = megapacdf['dps'].apply(plotting_label)
megapwrspdf['plotlabel'] = megapwrspdf['dps'].apply(plotting_label)


# %%

def rand_jitter(arr):
    return arr + (np.random.random(len(arr))-0.5) * 0.25

# %%
plotcounter = 0
colourchoice = 0
#might have to account for subtime, remember to check

powspecbands = ['δ', 'θ', 'α', 'β', 'low-γ', 'high-γ']

# %%
fonti = 16


#plotting LFPs for each day 
plt.close('all')
for date in megapwrspdf['date'].unique():
    fig, ax = plt.subplots(2,1, figsize = (15, 12))
    for hemi in ['ipsi', 'contra']:
        if hemi == 'ipsi':
            colour2use = ['#004d34', '#36c27c']
        elif hemi == 'contra':
            colour2use = ['#69035d', '#c4458d']
         
        comcombinedLFP = []
        checkvalid = megapwrspdf[(megapwrspdf['date'] == date) & (megapwrspdf['stim'] == 'S')& 
                                 (megapwrspdf['resampled'] == False) & (megapwrspdf['hemi'] == hemi)].reset_index(drop = True)
        if len(checkvalid) > 0:
            for elec in megapwrspdf['electrode'].unique():
                touse = megapwrspdf[(megapwrspdf['date'] == date) & (megapwrspdf['stim'] == 'S')&(megapwrspdf['electrode'] == elec) & 
                                (megapwrspdf['resampled'] == False) & (megapwrspdf['hemi'] == hemi)].reset_index(drop = True)
                combinedLFP = []
                for i in range(len(touse)):
                    freq2plot = touse.loc[i]['freq'][np.where(touse.loc[i]['freq'] <= 140)[0]]
                    pwrsp2plot = np.mean(touse.loc[i]['pwrsp'], axis=0)[np.where(touse.loc[i]['freq'] <= 140)[0]]
                    freqtodrop = np.where((freq2plot >= 59)& (freq2plot <= 61))[0]
                    freq2plot = np.delete(freq2plot,freqtodrop)
                    pwrsp2plot = np.delete(pwrsp2plot,freqtodrop)
                    freqtodrop = np.where((freq2plot >= 119)& (freq2plot <= 121))[0]
                    freq2plot = np.delete(freq2plot,freqtodrop)
                    pwrsp2plot = np.delete(pwrsp2plot,freqtodrop)
                    combinedLFP.append(pwrsp2plot)
                combinedLFP = np.mean(np.array(combinedLFP), axis = 0)
                comcombinedLFP.append(combinedLFP)
                ax[0].plot(freq2plot,combinedLFP, color = colour2use[0], alpha = 0.1, linewidth = 3)
            comcombinedLFP = np.mean(np.array(comcombinedLFP), axis = 0)
            ax[0].plot(freq2plot, comcombinedLFP, color = colour2use[1], alpha = 1, linewidth = 2)
        else:
            print('No data for ' + date + ' stim')

        checkvalid = megapwrspdf[(megapwrspdf['date'] == date) & (megapwrspdf['stim'] == 'C')& 
                                 (megapwrspdf['resampled'] == False) & (megapwrspdf['hemi'] == hemi)].reset_index(drop = True)
        if len(checkvalid) > 0:
            comcombinedLFP = []
            for elec in megapwrspdf['electrode'].unique():
                touse = megapwrspdf[(megapwrspdf['date'] == date) & (megapwrspdf['stim'] == 'C')&(megapwrspdf['electrode'] == elec) & 
                                (megapwrspdf['resampled'] == False) & (megapwrspdf['hemi'] ==  hemi)].reset_index(drop = True)
                combinedLFP = [] 
                for i in range(len(touse)):
                    freq2plot = touse.loc[i]['freq'][np.where(touse.loc[i]['freq'] <= 140)[0]]
                    pwrsp2plot = np.mean(touse.loc[i]['pwrsp'], axis=0)[np.where(touse.loc[i]['freq'] <= 140)[0]]
                    freqtodrop = np.where((freq2plot >= 59)& (freq2plot <= 61))[0]
                    freq2plot = np.delete(freq2plot,freqtodrop)
                    pwrsp2plot = np.delete(pwrsp2plot,freqtodrop)
                    freqtodrop = np.where((freq2plot >= 119)& (freq2plot <= 121))[0]
                    freq2plot = np.delete(freq2plot,freqtodrop)
                    pwrsp2plot = np.delete(pwrsp2plot,freqtodrop)
                    combinedLFP.append(pwrsp2plot)
                combinedLFP = np.mean(np.array(combinedLFP), axis = 0)
                comcombinedLFP.append(combinedLFP)
                ax[1].plot(freq2plot,combinedLFP, color = colour2use[0], alpha = 0.1, linewidth = 3)
            comcombinedLFP = np.mean(np.array(comcombinedLFP), axis = 0)
            ax[1].plot(freq2plot, comcombinedLFP, color = colour2use[1], alpha = 1, linewidth = 2)
        else:
            print('No data for ' + date + ' control')

    ax[0].set_title('stim', fontsize = fonti-2)
    ax[1].set_title('control', fontsize = fonti-2)
    ax[1].set_xlabel('Frequency (Hz)', fontsize = fonti-2)
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_tick_params(labelsize=fonti-2)
    ax[1].yaxis.set_tick_params(labelsize=fonti-2)
    ax[1].xaxis.set_tick_params(labelsize=fonti-2)
    ax[0].set_ylabel('Power (V²/Hz)', fontsize = fonti-2)
    ax[1].set_ylabel('Power (V²/Hz)', fontsize = fonti-2)
    ax[0].plot([], c='#c4458d', linestyle = 'solid', label='contra')
    ax[0].plot([], c='#36c27c', linestyle = 'solid', label='ipsi')
    ax[0].legend(loc='upper right', fontsize = fonti-2, frameon=False,handlelength=0, handletextpad=0, labelcolor='linecolor')
    fig.suptitle(date + ': ' +touse['dps'].unique()[0] + ' days post stim', fontsize = fonti+2)
    #ax.yaxis.grid(True, linestyle = '-', alpha = 0.2)
    sns.despine(top=True, right=True, left=False, bottom=False)


    plt.savefig('/home/bellvila/projects/rrg-bojana/bellvila/Kylie/plots/' +date + '_powspec_dropmice.png', bbox_inches='tight')
   
    plt.close('all')
    #ax.set_title(date + , fontsize = fonti)

      



#%%

plt.close('all')
for freq in range(len(powspecfrqs)-1):
    fig, ax = plt.subplots()
    vals = []
    for date in mouseinfodf['date'].unique():
        for electrode in megapwrspdf[megapwrspdf['resampled'] == False]['electrode'].unique():
            for stim in megapwrspdf['stim'].unique():
                tolook = megapwrspdf[(megapwrspdf['resampled'] == False)&(megapwrspdf['date'] == date)&
                             (megapwrspdf['electrode'] == electrode)&
                             (megapwrspdf['stim'] == stim)]
                
                tolook1 = tolook[tolook['hemi'] == 'ipsi'].reset_index(drop = True)
                avpwr = []
                for i in range(len(tolook1)):
                    freqtemp = tolook1.loc[i]['freq']
                    pwrsptemp = np.mean(tolook1.loc[i]['pwrsp'], axis=0)
                    freqtodrop = np.where((freqtemp >= 59)& (freqtemp <= 61))[0]
                    freqtemp = np.delete(freqtemp,freqtodrop)
                    pwrsptemp = np.delete(pwrsptemp,freqtodrop)
                    freqtodrop = np.where((freqtemp >= 119)& (freqtemp <= 121))[0]
                    freqtemp = np.delete(freqtemp,freqtodrop)
                    pwrsptemp = np.delete(pwrsptemp,freqtodrop)
                    avpwr.append(pwrsptemp[np.where((freqtemp >= powspecfrqs[freq])& (freqtemp < powspecfrqs[freq+1]))[0]])
                    freqi = freqtemp[np.where((freqtemp >= powspecfrqs[freq])& (freqtemp < powspecfrqs[freq+1]))[0]]
                avpwripsi = np.mean(np.array(avpwr))
                
                tolook2 = tolook[tolook['hemi'] == 'contra'].reset_index(drop = True)
                avpwr = []
                for i in range(len(tolook2)):
                    freqtemp = tolook2.loc[i]['freq']
                    pwrsptemp = np.mean(tolook2.loc[i]['pwrsp'], axis=0)
                    freqtodrop = np.where((freqtemp >= 59)& (freqtemp <= 61))[0]
                    freqtemp = np.delete(freqtemp,freqtodrop)
                    pwrsptemp = np.delete(pwrsptemp,freqtodrop)
                    freqtodrop = np.where((freqtemp >= 119)& (freqtemp <= 121))[0]
                    freqtemp = np.delete(freqtemp,freqtodrop)
                    pwrsptemp = np.delete(pwrsptemp,freqtodrop)
                    avpwr.append(pwrsptemp[np.where((freqtemp >= powspecfrqs[freq])& (freqtemp < powspecfrqs[freq+1]))[0]])
                    freqi = freqtemp[np.where((freqtemp >= powspecfrqs[freq])& (freqtemp < powspecfrqs[freq+1]))[0]]
                avpwrcontra = np.mean(np.array(avpwr))
                if (len(tolook) > 0) & (len(tolook1) > 0) & (len(tolook2)> 0):
                    vals.append([electrode, stim, avpwripsi/avpwrcontra, tolook['plotpos'].unique()[0], tolook['plotlabel'].unique()[0]])
            
    valsdf = pd.DataFrame(vals, columns = ['electrode', 'stim', 'ipsi/contra', 'plotpos', 'plotlabel'])
    bp1 = valsdf[valsdf['stim'] == 'C']
    bp2 = valsdf[valsdf['stim'] == 'S']
    
    
    box1 = ax.boxplot(bp1.groupby(['plotpos'])['ipsi/contra'].apply(list), positions = np.array(sorted(bp1['plotpos'].unique()))-0.3, widths = 0.5, showfliers=False)
    bpcol(box1, colourscontrol[1])
    ax.scatter(rand_jitter(bp1['plotpos']) - 0.3, bp1['ipsi/contra'], color = colourscontrol[1], alpha = 0.6)
    
    box2 = ax.boxplot(bp2.groupby(['plotpos'])['ipsi/contra'].apply(list), positions = np.array(sorted(bp2['plotpos'].unique()))+0.3, widths = 0.5, showfliers=False)
    bpcol(box2, coloursstim[1])
    ax.scatter(rand_jitter(bp2['plotpos'])+ 0.3, bp2['ipsi/contra'], color = coloursstim[1], alpha = 0.6)
    
    ax.set_xticks(megapwrspdf['plotpos'].unique(), labels = megapwrspdf['plotlabel'].unique())
    #ax.set_ylim([0, 2])
    ax.yaxis.set_tick_params(labelsize=fonti-2)
    ax.xaxis.set_tick_params(labelsize=fonti-2)
    ax.set_xlabel('Days post stroke', fontsize = fonti)
    ax.set_ylabel('Ipsilesional/Contralesional mean power', fontsize = fonti)
    ax.set_title(powspecbands[freq] + ' frequency band', fontsize = fonti)
    ax.plot([], c=coloursstim[1], linestyle = 'solid', label='stim')
    ax.plot([], c=colourscontrol[1], linestyle = 'solid', label='control')
    if freq == 0:
        ax.legend(loc='upper right', fontsize = fonti-2, frameon=False,handlelength=0, handletextpad=0, labelcolor='linecolor')
    ax.yaxis.grid(True, linestyle = '-', alpha = 0.2)
    sns.despine(top=True, right=True, left=False, bottom=False)
    #ax.spines[['right', 'top']].set_visibile(False)
    
    
    plt.savefig('/home/bellvila/projects/rrg-bojana/bellvila/Kylie/plots/' +powspecbands[freq] + '_frac_nice_dropmice.png', bbox_inches='tight')
   
    plt.close('all')
    
    plotcounter += 1
    
    


# %%


plt.close('all')
for freq in range(len(powspecfrqs)-1):
    fig, ax = plt.subplots()
    vals = []
    for date in mouseinfodf['date'].unique():
        for electrode in megapwrspdf[megapwrspdf['resampled'] == False]['electrode'].unique():
            for stim in megapwrspdf['stim'].unique():
                tolook = megapwrspdf[(megapwrspdf['resampled'] == False)&(megapwrspdf['date'] == date)&
                             (megapwrspdf['electrode'] == electrode)&
                             (megapwrspdf['stim'] == stim)]
                
                tolook1 = tolook[tolook['hemi'] == 'ipsi'].reset_index(drop = True)
                avpwr = []
                for i in range(len(tolook1)):
                    freqtemp = tolook1.loc[i]['freq']
                    pwrsptemp = np.mean(tolook1.loc[i]['pwrsp'], axis=0)
                    freqtodrop = np.where((freqtemp >= 59)& (freqtemp <= 61))[0]
                    freqtemp = np.delete(freqtemp,freqtodrop)
                    pwrsptemp = np.delete(pwrsptemp,freqtodrop)
                    freqtodrop = np.where((freqtemp >= 119)& (freqtemp <= 121))[0]
                    freqtemp = np.delete(freqtemp,freqtodrop)
                    pwrsptemp = np.delete(pwrsptemp,freqtodrop)
                    avpwr.append(pwrsptemp[np.where((freqtemp >= powspecfrqs[freq])& (freqtemp < powspecfrqs[freq+1]))[0]])
                    freqi = freqtemp[np.where((freqtemp >= powspecfrqs[freq])& (freqtemp < powspecfrqs[freq+1]))[0]]
                avpwripsi = np.mean(np.array(avpwr))
                
                tolook2 = tolook[tolook['hemi'] == 'contra'].reset_index(drop = True)
                avpwr = []
                for i in range(len(tolook2)):
                    freqtemp = tolook2.loc[i]['freq']
                    pwrsptemp = np.mean(tolook2.loc[i]['pwrsp'], axis=0)
                    freqtodrop = np.where((freqtemp >= 59)& (freqtemp <= 61))[0]
                    freqtemp = np.delete(freqtemp,freqtodrop)
                    pwrsptemp = np.delete(pwrsptemp,freqtodrop)
                    freqtodrop = np.where((freqtemp >= 119)& (freqtemp <= 121))[0]
                    freqtemp = np.delete(freqtemp,freqtodrop)
                    pwrsptemp = np.delete(pwrsptemp,freqtodrop)
                    avpwr.append(pwrsptemp[np.where((freqtemp >= powspecfrqs[freq])& (freqtemp < powspecfrqs[freq+1]))[0]])
                    freqi = freqtemp[np.where((freqtemp >= powspecfrqs[freq])& (freqtemp < powspecfrqs[freq+1]))[0]]
                avpwrcontra = np.mean(np.array(avpwr))
                if (len(tolook) > 0) & (len(tolook1) > 0) & (len(tolook2)> 0):
                    vals.append([date, electrode, stim, avpwripsi/avpwrcontra, tolook['plotpos'].unique()[0], tolook['plotlabel'].unique()[0]])
            
    valsdf = pd.DataFrame(vals, columns = ['date','electrode', 'stim', 'ipsi/contra', 'plotpos', 'plotlabel'])
    bp1 = valsdf[valsdf['stim'] == 'C']
    bp2 = valsdf[valsdf['stim'] == 'S']
    
    bp1 = bp1.groupby(['date', 'plotpos'])['ipsi/contra'].mean().reset_index()
    bp2 = bp2.groupby(['date', 'plotpos'])['ipsi/contra'].mean().reset_index()
    
    box1 = ax.boxplot(bp1.groupby(['plotpos'])['ipsi/contra'].apply(list), positions = np.array(sorted(bp1['plotpos'].unique()))-0.3, widths = 0.5, showfliers=False)
    bpcol(box1, colourscontrol[1])
    ax.scatter(rand_jitter(bp1['plotpos']) - 0.3, bp1['ipsi/contra'], color = colourscontrol[1], alpha = 0.6)
    
    box2 = ax.boxplot(bp2.groupby(['plotpos'])['ipsi/contra'].apply(list), positions = np.array(sorted(bp2['plotpos'].unique()))+0.3, widths = 0.5, showfliers=False)
    bpcol(box2, coloursstim[1])
    ax.scatter(rand_jitter(bp2['plotpos'])+ 0.3, bp2['ipsi/contra'], color = coloursstim[1], alpha = 0.6)
    
    ax.set_xticks(megapwrspdf['plotpos'].unique(), labels = megapwrspdf['plotlabel'].unique())
    #ax.set_ylim([0, 2])
    ax.yaxis.set_tick_params(labelsize=fonti-2)
    ax.xaxis.set_tick_params(labelsize=fonti-2)
    ax.set_xlabel('Days post stroke', fontsize = fonti)
    ax.set_ylabel('Ipsilesional/Contralesional mean power', fontsize = fonti)
    ax.set_title(powspecbands[freq] + ' frequency band: Mouse averaged', fontsize = fonti)
    ax.plot([], c=coloursstim[1], linestyle = 'solid', label='stim')
    ax.plot([], c=colourscontrol[1], linestyle = 'solid', label='control')
    if freq == 0:
        ax.legend(loc='upper right', fontsize = fonti-2, frameon=False,handlelength=0, handletextpad=0, labelcolor='linecolor')
    ax.yaxis.grid(True, linestyle = '-', alpha = 0.2)
    sns.despine(top=True, right=True, left=False, bottom=False)
    #ax.spines[['right', 'top']].set_visibile(False)
    
    
    plt.savefig('/home/bellvila/projects/rrg-bojana/bellvila/Kylie/plots/' +powspecbands[freq] + '_frac_mouseavg_dropmice.png', bbox_inches='tight')
   
    plt.close('all')
    
    plotcounter += 1


    
#%%
for i in range(len(megapacdf)):
    p = megapacdf.loc[i]['p']
    MI = megapacdf.loc[i]['xpac']
    phases = p.xvec  
    amplitudes = p.yvec
    surro = p.surrogates.squeeze()
    s_mean, s_std = np.mean(surro, axis=0), np.std(surro, axis=0)
    thresh = scipy.stats.norm.ppf(1-pval,s_mean,s_std)
    MI_pval = MI*np.where(MI >= thresh, 1, 0)


    megapacdf.loc[i,'deltalowgamma'] = MI_pval[np.where(amplitudes < highfrq)[0]][:,np.where(phases <= lowfrqs[0])[0]].mean()
    megapacdf.loc[i, 'deltahighgamma'] =  MI_pval[np.where(amplitudes >= highfrq)[0]][:,np.where(phases <= lowfrqs[0])[0]].mean()
    megapacdf.loc[i,'thetalowgamma'] =  MI_pval[np.where(amplitudes < highfrq)[0]][:,np.where((phases >lowfrqs[0])&(phases <= lowfrqs[1]))[0]].mean()
    megapacdf.loc[i, 'thetahighgamma'] =  MI_pval[np.where(amplitudes >= highfrq)[0]][:,np.where((phases >lowfrqs[0])&(phases <= lowfrqs[1]))[0]].mean()
    megapacdf.loc[i,'alphalowgamma'] =  MI_pval[np.where(amplitudes < highfrq)[0]][:,np.where((phases >lowfrqs[1])&(phases <= lowfrqs[2]))[0]].mean()
    megapacdf.loc[i,'alphahighgamma'] =  MI_pval[np.where(amplitudes >= highfrq)[0]][:,np.where((phases >lowfrqs[1])&(phases <= lowfrqs[2]))[0]].mean()
    megapacdf.loc[i,'betalowgamma']  = MI_pval[np.where(amplitudes < highfrq)[0]][:,np.where(phases > lowfrqs[2])[0]].mean()
    megapacdf.loc[i,'betahighgamma']  = MI_pval[np.where(amplitudes >= highfrq)[0]][:,np.where(phases > lowfrqs[2])[0]].mean()
    


#%% 
for low in range(len(lows)):
    for high in range(len(highs)):
        fig, ax = plt.subplots()
        vals = []
        for date in mouseinfodf['date'].unique():
            for electrode in megapacdf[megapacdf['resampled'] == False]['electrode'].unique():
                for stim in megapacdf['stim'].unique():
                    tolook = megapacdf[(megapacdf['resampled'] == False)&(megapacdf['date'] == date)&
                                 (megapacdf['electrode'] == electrode)&
                                 (megapacdf['stim'] == stim)&
                                 (megapacdf['pacmethod'] == 'tort')]
                    if len(tolook) > 0:
                        tolook1 = tolook[tolook['hemi'] == 'ipsi'].reset_index(drop = True)
                        avMIipsi = tolook1[lowlows[low] + highhighs[high]].mean()
                    
                    
                        tolook2 = tolook[tolook['hemi'] == 'contra'].reset_index(drop = True)
                        avMIcontra = tolook2[lowlows[low] + highhighs[high]].mean()
                    
                    
                    
                        vals.append([electrode, stim, avMIipsi/avMIcontra,tolook['plotpos'].unique()[0], tolook['plotlabel'].unique()[0]])
                
        valsdf = pd.DataFrame(vals, columns = ['electrode', 'stim', 'ipsi/contra', 'plotpos', 'plotlabel'])
        bp1 = valsdf[valsdf['stim'] == 'C']
        bp2 = valsdf[valsdf['stim'] == 'S']
        
        box1 = ax.boxplot(bp1.groupby(['plotpos'])['ipsi/contra'].apply(list), positions = np.array(sorted(bp1['plotpos'].unique()))-0.15, widths = 0.3, showfliers=False)
        bpcol(box1, colourscontrol[1])
        ax.scatter(rand_jitter(bp1['plotpos']) - 0.15, bp1['ipsi/contra'], color = colourscontrol[1], alpha = 0.6)
        
        box2 = ax.boxplot(bp2.groupby(['plotpos'])['ipsi/contra'].apply(list), positions = np.array(sorted(bp2['plotpos'].unique()))+0.15, widths = 0.3, showfliers=False)
        bpcol(box2, coloursstim[1])
        ax.scatter(rand_jitter(bp2['plotpos'])+ 0.15, bp2['ipsi/contra'], color = coloursstim[1], alpha = 0.6)
        
        ax.set_xticks(megapwrspdf['plotpos'].unique(), labels = megapwrspdf['plotlabel'].unique())
        ax.set_ylim([-0.5, 5])
        ax.yaxis.set_tick_params(labelsize=fonti-2)
        ax.xaxis.set_tick_params(labelsize=fonti-2)
        ax.set_xlabel('Days post stroke', fontsize = fonti)
        ax.set_ylabel('Ipsilesional/Contralesional MI', fontsize = fonti)
        ax.set_title(lows[low] + highs[high] + ' coupling', fontsize = fonti)
        ax.plot([], c=coloursstim[1], linestyle = 'solid', label='stim')
        ax.plot([], c=colourscontrol[1], linestyle = 'solid', label='control')
        
        ax.legend(loc='upper right', fontsize = fonti-2, frameon=False,handlelength=0, handletextpad=0, labelcolor='linecolor')
        #ax.spines[['right', 'top']].set_visibile(False)
        ax.yaxis.grid(True, linestyle = '-', alpha = 0.2)
        sns.despine(top=True, right=True, left=False, bottom=False)
        
        
        plt.savefig('/home/bellvila/projects/rrg-bojana/bellvila/Kylie/plots/pacboxplot' + lowlows[low] + highhighs[high] + '_frac_nice_scaled_dropmice.png', bbox_inches='tight')
       
        plt.close('all')
        
        plotcounter += 1
            
            
# %%

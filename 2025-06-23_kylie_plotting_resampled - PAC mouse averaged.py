## PLOTTING CODE!! 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import scipy.stats
import os

pval = 0.05


#loading the unresampled pickles
megapwrspdf = pd.read_pickle("/home/achanna/projects/rrg-bojana/achanna/megapwrspdf_resampled.pkl")
megapacdf   = pd.read_pickle("/home/achanna/projects/rrg-bojana/achanna/megapacdf_resampled.pkl")


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


#output directory (save location) for plots
outdir = "/home/achanna/projects/rrg-bojana/achanna/kylie_edits/resampled_plots/"


#dropping if outside 20minutes
indexAge = megapwrspdf[megapwrspdf['tp']*2 >= 20].index
megapwrspdf.drop(indexAge , inplace=True)
megapwrspdf = megapwrspdf.reset_index(drop = True)
indexAge = megapacdf[megapacdf['pacwintp'] > 10].index
megapacdf.drop(indexAge , inplace=True)
megapacdf = megapacdf.reset_index(drop = True)

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


# Perform a left merge with indicator to identify rows to drop
merged = megapwrspdf.merge(micetoexclude, on=['date', 'stim'], how='left', indicator=True)
# Keep only rows that were not in the todrop dataframe
megapwrspdf = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
megapwrspdf = megapwrspdf.reset_index(drop = True)

merged =  megapacdf.merge(micetoexclude, on=['date', 'stim'], how='left', indicator=True)
megapacdf = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
megapacdf = megapacdf.reset_index(drop = True)


#plotting time (copied from Monica's original code)
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

""" 
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
                                 (megapwrspdf['resampled'] == True) & (megapwrspdf['hemi'] == hemi)].reset_index(drop = True)
        if len(checkvalid) > 0:
            for elec in megapwrspdf['electrode'].unique():
                touse = megapwrspdf[(megapwrspdf['date'] == date) & (megapwrspdf['stim'] == 'S')&(megapwrspdf['electrode'] == elec) & 
                                (megapwrspdf['resampled'] == True) & (megapwrspdf['hemi'] == hemi)].reset_index(drop = True)
                if len(touse) == 0:
                    continue
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
                                 (megapwrspdf['resampled'] == True) & (megapwrspdf['hemi'] == hemi)].reset_index(drop = True)
        if len(checkvalid) > 0:
            comcombinedLFP = []
            for elec in megapwrspdf['electrode'].unique():
                touse = megapwrspdf[(megapwrspdf['date'] == date) & (megapwrspdf['stim'] == 'C')&(megapwrspdf['electrode'] == elec) & 
                                (megapwrspdf['resampled'] == True) & (megapwrspdf['hemi'] ==  hemi)].reset_index(drop = True)
                if len(touse) == 0:
                    continue
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


    plt.savefig('/home/achanna/projects/rrg-bojana/achanna/kylie_edits/resampled_plots/' +date + '_powspec_dropmice.png', bbox_inches='tight')
   
    plt.close('all')
    #ax.set_title(date + , fontsize = fonti)

   """    


""" 
#%%

plt.close('all')
for freq in range(len(powspecfrqs)-1):
    fig, ax = plt.subplots()
    vals = []
    for date in megapwrspdf['date'].unique():
        for electrode in megapwrspdf[megapwrspdf['resampled'] == True]['electrode'].unique():
            for stim in megapwrspdf['stim'].unique():
                tolook = megapwrspdf[(megapwrspdf['resampled'] == True)&(megapwrspdf['date'] == date)&
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
    
    
    plt.savefig('/home/achanna/projects/rrg-bojana/achanna/kylie_edits/resampled_plots/' +powspecbands[freq] + '_frac_nice_dropmice.png', bbox_inches='tight')
   
    plt.close('all')
    
    plotcounter += 1
    """ 
    

""" 
# %%


plt.close('all')
for freq in range(len(powspecfrqs)-1):
    fig, ax = plt.subplots()
    vals = []
    for date in megapwrspdf['date'].unique():
        for electrode in megapwrspdf[megapwrspdf['resampled'] == True]['electrode'].unique():
            for stim in megapwrspdf['stim'].unique():
                tolook = megapwrspdf[(megapwrspdf['resampled'] == True)&(megapwrspdf['date'] == date)&
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
    
    
    plt.savefig('/home/achanna/projects/rrg-bojana/achanna/kylie_edits/resampled_plots/' +powspecbands[freq] + '_frac_mouseavg_dropmice.png', bbox_inches='tight')
   
    plt.close('all')
    
    plotcounter += 1
 """

    
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
        for date in megapacdf['date'].unique():
            for electrode in megapacdf[megapacdf['resampled'] == True]['electrode'].unique():
                for stim in megapacdf['stim'].unique():
                    tolook = megapacdf[(megapacdf['resampled'] == True)&(megapacdf['date'] == date)&
                                 (megapacdf['electrode'] == electrode)&
                                 (megapacdf['stim'] == stim)&
                                 (megapacdf['pacmethod'] == 'tort')]
                    if len(tolook) > 0:
                        tolook1 = tolook[tolook['hemi'] == 'ipsi'].reset_index(drop = True)
                        avMIipsi = tolook1[lowlows[low] + highhighs[high]].mean()
                    
                    
                        tolook2 = tolook[tolook['hemi'] == 'contra'].reset_index(drop = True)
                        avMIcontra = tolook2[lowlows[low] + highhighs[high]].mean()
                    
                    
                    
                        vals.append([date, electrode, stim, avMIipsi/avMIcontra,tolook['plotpos'].unique()[0], tolook['plotlabel'].unique()[0]])    #NEW: added 'date'
                
        valsdf = pd.DataFrame(vals, columns = ['date', 'electrode', 'stim', 'ipsi/contra', 'plotpos', 'plotlabel'])                                 #NEW: added 'date'
        bp1 = valsdf[valsdf['stim'] == 'C']
        bp2 = valsdf[valsdf['stim'] == 'S']
        
        #NEW: mouse-average across electrodes within each date
        bp1 = bp1.groupby(['date','plotpos'])['ipsi/contra'].mean().reset_index()
        bp2 = bp2.groupby(['date','plotpos'])['ipsi/contra'].mean().reset_index()

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
        ax.set_ylabel('Ipsilesional/Contralesional MI: Mouse averaged', fontsize = fonti)          #NEW: modified vertical axis to show "Mouse averaged"
        ax.set_title(lows[low] + highs[high] + ' coupling', fontsize = fonti)
        ax.plot([], c=coloursstim[1], linestyle = 'solid', label='stim')
        ax.plot([], c=colourscontrol[1], linestyle = 'solid', label='control')
        
        ax.legend(loc='upper right', fontsize = fonti-2, frameon=False,handlelength=0, handletextpad=0, labelcolor='linecolor')
        #ax.spines[['right', 'top']].set_visibile(False)
        ax.yaxis.grid(True, linestyle = '-', alpha = 0.2)
        sns.despine(top=True, right=True, left=False, bottom=False)
        
        
        plt.savefig('/home/achanna/projects/rrg-bojana/achanna/kylie_edits/resampled_plots/' + lowlows[low] + highhighs[high] + '_frac_mouseavg_scaled_dropmice.png', bbox_inches='tight')
       
        plt.close('all')
        
        plotcounter += 1


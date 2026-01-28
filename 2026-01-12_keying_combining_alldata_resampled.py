#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:49:00 2024

@author: mbellvila

MODIFIED MONICA'S CODE FOR KEYING on Tue Jan 20, 2025
"""

# COMBINING CODE FOR RESAMPLED!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import scipy.stats
import pickle #forgot to import pickle earlier?

pval = 0.05


# #already ran data (this is how each looks after: /home/achanna/projects/rrg-bojana/bellvila/Kylie/2024-02-01/)
# directories_old = sorted(glob.glob('/home/achanna/projects/rrg-bojana/bellvila/Kylie/202*/'))
# directories_old = directories_old[:-4] #drop last 4

# #newly run data (this is how each looks after: /home/achanna/projects/rrg-bojana/achanna/2025-10-10_)
# empty = []
# for pwr_file in sorted(glob.glob('/home/achanna/projects/rrg-bojana/achanna/2025-*pwrsp_dec.pickle')):
#     x = pwr_file.replace('_pwrsp_dec.pickle', '_')
#     pac_file = x + 'pac_dec.pickle'
#     empty.append(x)                             #eg. 2025-10-10_pac_dec.pickle
# directories_new = empty

# #combine
# directories = directories_old + directories_new



#collecting all data for keying
empty = []
for pwr_file in sorted(glob.glob('/lustre06/project/6061907/achanna/keying_edits/resampled/*_resampledLFP_pwrsp_dec.pickle')):
    x = pwr_file.replace('pwrsp_dec.pickle','')   
    empty.append(x)

directories = empty


#wrote this out for keying from her experiment notes
mouseinfo = [
    # Animal 1: Req No.8658 (2-1R)
    ['2025-12-04','130747','ipsi',  'UNKNOWN', 0, 0,'F'],
    ['2025-12-04','140801','contra','UNKNOWN', 0, 0,'F'],
    # Animal 2: Req No.8658 (3-2L)
    ['2025-12-04','161031','ipsi',  'UNKNOWN', 0, 0,'F'],
    ['2025-12-04','165814','contra','UNKNOWN', 0, 0,'F'],
    
    # Animal 1: Cage C (Mouse 1)
    ['2025-12-09','121244','ipsi',  'C', 0, 0,'F'],
    ['2025-12-09','130236','contra','C', 0, 0,'F'],
    # Animal 2: Cage B (Mouse 2)
    ['2025-12-09','150826','ipsi',  'B', 0, 0,'F'],
    ['2025-12-09','155851','contra','B', 0, 0,'F'],

    # Animal 1: Cage C (Mouse 2)
    ['2025-12-10','125739','ipsi',  'C', 0, 0,'F'],
    ['2025-12-10','134421','contra','C', 0, 0,'F'],
    # Animal 2: Cage B (Mouse 1)
    ['2025-12-10','160810','ipsi',  'B', 0, 0,'F'],
    ['2025-12-10','165732','contra','B', 0, 0,'F'],

    # Animal 1: Cage C (Mouse 3)
    ['2025-12-11','121422','ipsi',  'C', 0, 0,'F'],
    ['2025-12-11','130339','contra','C', 0, 0,'F'],
    # Animal 2: Cage B (Mouse 3)
    ['2025-12-11','151206','ipsi',  'B', 0, 0,'F'],
    ['2025-12-11','155502','contra','B', 0, 0,'F'],

    # Animal 1: Cage A (Mouse 1)
    ['2025-12-12','160345','ipsi',  'A', 0, 0,'F'],
    ['2025-12-12','165448','contra','A', 0, 0,'F'],
]

#deleted monica's table vvv
""" mouseinfo = [['2024-02-01', '134503', 'ipsi', 'C', 1, '5', 'M'], 
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
             ['2025-10-08', '123509', 'contra', 'C', 2, '6', 'M'], 	#2025-10-08
	     ['2025-10-08', '133902', 'ipsi', 'C', 2, '6', 'M'],
	     ['2025-10-08', '160356', 'contra', 'S', 2, '6', 'F'],
         ['2025-10-08', '165330', 'ipsi', 'S', 2, '6', 'F'],
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
             ] """

#NEW: changed 'stim' to 'cage'
mouseinfodf = pd.DataFrame(mouseinfo, columns= ['date', 'subtime', 'hemisphere', 'cage', 'dayspostarrival','dpstim', 'sex'])

megapwrspdf = pd.DataFrame()
megapacdf = pd.DataFrame()




# %%


# import os

for directory in directories:

    # if os.path.isdir(directory):
    #     #old data (Monica folders in Kylie/2024-xx-xx/)
    #     pwr_path = os.path.join(directory, 'pwrsp_dec')
    #     pac_path = os.path.join(directory, 'pac_dec')

    #     #old data is numpy
    #     pwrspvals = np.load(pwr_path, allow_pickle=True)
    #     pacvals   = np.load(pac_path, allow_pickle=True)
    # else:
    #     #new data (Ammar files in achanna/2025-xx-xx/)
    #     pwr_path = directory + 'pwrsp_dec.pickle'
    #     pac_path = directory + 'pac_dec.pickle'

    #     #new data is pickle
    #     with open(pwr_path, 'rb') as f:
    #         pwrspvals = pickle.load(f)
    #     with open(pac_path, 'rb') as f:
    #         pacvals = pickle.load(f)

    pwr_path = directory + "pwrsp_dec.pickle"
    pac_path = directory + "pac_dec.pickle"

    with open(pwr_path, 'rb') as f:
        pwrspvals = pickle.load(f)
    with open(pac_path, 'rb') as f:
        pacvals = pickle.load(f)
    
    #everything below in this for loop is UNCHANGED
    pwrspdf = pd.DataFrame(pwrspvals, columns = ['date', 'subtime', 'electrode', 'totdur', 'tp', 'pwrspwin', 'Fs', 'freq', 'pwrsp','resampled'])                                           
    pacdf = pd.DataFrame(pacvals, columns = ['date', 'subtime', 'electrode', 'tortdur', 'tp', 'pacwinsize', 'pacwintp', 'Fs', 'pacmethod', 'resampled','p', 'xpac', 'pval', 'xpac_smean'])

    for i in range(len(pwrspdf)):
        mousetouse = mouseinfodf[(mouseinfodf['date'] == pwrspdf.loc[i]['date'])&(mouseinfodf['subtime'] == pwrspdf.loc[i]['subtime'])]
        pwrspdf.loc[i,'hemi'] = mousetouse['hemisphere'].item()
        pwrspdf.loc[i,'cage'] = mousetouse['cage'].item()
        pwrspdf.loc[i,'dpa'] = mousetouse['dayspostarrival'].item() 
        pwrspdf.loc[i,'dps'] = mousetouse['dpstim'].item() 
        pwrspdf.loc[i, 'sex'] =  mousetouse['sex'].item() 


    for i in range(len(pacdf)):
        mousetouse = mouseinfodf[(mouseinfodf['date'] == pacdf.loc[i]['date'])&(mouseinfodf['subtime'] == pacdf.loc[i]['subtime'])]
        pacdf.loc[i,'hemi'] = mousetouse['hemisphere'].item()
        pacdf.loc[i,'cage'] = mousetouse['cage'].item()
        pacdf.loc[i,'dpa'] = mousetouse['dayspostarrival'].item()
        pacdf.loc[i, 'dps'] = mousetouse['dpstim'].item() 
        pacdf.loc[i, 'sex'] =  mousetouse['sex'].item() 
    megapwrspdf = pd.concat((megapwrspdf, pwrspdf), ignore_index = True)
    megapacdf = pd.concat((megapacdf, pacdf), ignore_index = True)

megapwrspdf = megapwrspdf[megapwrspdf['resampled'] == False].reset_index(drop=True)
megapacdf   = megapacdf[megapacdf['resampled'] == False].reset_index(drop=True)


# #UPDATED for combining code: Channels 0 and 15 are empty ONLY for Monica's old 2024 data (folder-based recordings)
# old_pwr = megapwrspdf['date'].astype(str).str.startswith('2024')
# old_pac = megapacdf['date'].astype(str).str.startswith('2024')


# #--PWRSP-- (untouched)
# indexelectrode = megapwrspdf[(megapwrspdf['electrode'] > 14) & old_pwr].index       #drop electrode 15+ only for old pwrsp data
# megapwrspdf.drop(indexelectrode, inplace=True)

# indexelectrode = megapwrspdf[(megapwrspdf['electrode'] < 1) & old_pwr].index        #drop electrode 0 only for old data
# megapwrspdf.drop(indexelectrode, inplace=True)

# megapwrspdf = megapwrspdf.reset_index(drop=True)
# #--PAC-- (untouched)
# indexelectrode = megapacdf[(megapacdf['electrode'] > 14) & old_pac].index           #drop electrode 15+ only for old pac 
# megapacdf.drop(indexelectrode, inplace=True)

# indexelectrode = megapacdf[(megapacdf['electrode'] < 1) & old_pac].index            #drop electrode 0 only for old pac 
# megapacdf.drop(indexelectrode, inplace=True)

# megapacdf = megapacdf.reset_index(drop=True)


#%%
megapwrspdf.to_pickle("/lustre06/project/6061907/achanna/keying_edits/resampled/megapwrspdf_keying_resampled.pkl")
megapacdf.to_pickle("/lustre06/project/6061907/achanna/keying_edits/resampled/megapacdf_keying_resampled.pkl")

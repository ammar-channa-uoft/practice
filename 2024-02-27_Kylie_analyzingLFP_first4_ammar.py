#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:19:25 2024

@author: mbellvila
"""
#newly ran data 
#changed nomenclature due to forepaw collected seperately and mouse name was different

import numpy as np
import scipy.io
import h5py
import mat73
from scipy.signal import hilbert
import pickle
import glob
import tdt
from scipy import signal
from tensorpac import Pac
from tensorpac.signals import pac_signals_tort
import scipy as sc
from scipy.fft import fft as fft 
from multiprocessing import Pool

#selecting the 4 newest experiments
directories = sorted(glob.glob('/home/achanna/projects/rrg-bojana/bellvila/Kylie/202*/'), reverse = True)[:4]

f_amp = (30, 140, 5, 1)
f_pha = (1, 29, 2, .5)
n_perm = 200
pwrspwin = 2 #window for power spec calc
totdur = 2*60
pacwindow = 10 #window for pac calc
n_jobs = 18
dec = 10



def nextpow2(x):
    """returns the smallest power of two that is greater than or equal to the
    absolute value of x.

    This function is useful for optimizing FFT operations, which are
    most efficient when sequence length is an exact power of two.

    :Example:

    .. doctest::

        >>> from spectrum import nextpow2
        >>> x = [255, 256, 257]
        >>> nextpow2(x)
        array([8, 8, 9])

    """
    res = np.ceil(np.log2(x))
    return res.astype('int')  #we want integer values only but ceil gives float

def powerspeccalc(window, ephysdata, samplefreq):
    npoints = len(ephysdata)
    nsegment = int(round(npoints/window)-1)
    NFFT = 2**(nextpow2(npoints/nsegment))
    spec = np.zeros([nsegment, NFFT])
    
    for ii in range(nsegment):
        #Split up section into subsections
        tempval = ii*window
        fftcalc = ephysdata[int(tempval):int(tempval+window-1)]
         #Hanning window to reduce edging effects
        fftcalc = np.multiply(np.hanning(len(fftcalc)),fftcalc-np.mean(fftcalc))
        #Compute FFT at NFFT resolution
        spec[ii] = fft(fftcalc,NFFT)/npoints

    #compute frequency and power spectrum
    freq = samplefreq/2*np.linspace(0,1,int(NFFT/2)) #x-axis of power-spectrum
    pwrsp = 2 * abs(spec[:,0: int(len(spec[0])/2)])  #magnitude of FFT
    return freq, pwrsp

#loops over all experiment day folders
for directory in directories:
    print(directory)
    subdirectories = sorted(glob.glob(directory + 'Monica_Kylie_*/kyliemouse*')) #changing C57BL6J to kyliemouse (correct naming)
    date = directory[directory.find('202'):directory.find('202')+10] #Extracts date (eg. "2025-10-08") from the folder path string
    baseline_subdirectories = subdirectories[::2] #ensuring only baseline data is analyzed, not forepaw
    
    pwrspdata = [] #to store power spectrum results
    pacdata = []   #to store phase-amplitude coupling results
    for subdirectory in baseline_subdirectories:
        print(subdirectory)
        
        subtime = subdirectory[-6:] #extracts last 6 characters of folder's name
        try:
            data = tdt.read_block(subdirectory) #tries loading file
            run = 1 				#data loaded successfully
        except:
            try:
                data = tdt.read_block(subdirectory, t2 = 350) #if fails, tries loading file again but only first 350 secs
                run = 1 				      #data loaded successfully
            except:
                print(subdirectory + ' data could not be loaded')
                run = 0
        
        if run == 1: #continue only if data loaded
            raw = data.streams.raw_.data  #raw LFP signal
            Fs = data.streams.raw_.fs	  #sampling frequency for LFP signal
            
            stim = data.streams.MonA.data #forepaw stimulus signal
            stimfs = data.streams.MonA.fs #sampling frequency for forepaw stimulus signal
            
            try:
                firststimonset = np.where(abs(data.streams.MonA.data[0]) > 0)[0][0]/stimfs - 0.5    #time just before first stimulus
            except:
                firststimonset = len(raw[0])/Fs                                                     #if no stimulus, full signal kept
            

            #2nd-order low-pass filter w/ cutoff 140 Hz            
            b, a = signal.butter(2, 140, 'low',  analog=False, output='ba', fs=Fs)
            raw = raw[1:15] #dropping the noise/dead channels
            rawclip=raw[:,0:int(firststimonset*Fs)]
            rawclipresampled = rawclip[0::2,:] - rawclip[1::2,:]
            
            LFP = signal.filtfilt(b, a, rawclip)
            LFPresampled =  signal.filtfilt(b, a, rawclipresampled)
            
            #decimation
            fillLFP = []
            if dec > 1:    
                for i in range(len(LFP)):
                    fillLFP.append(sc.signal.decimate(LFP[i], dec, n=None, ftype='iir', axis=-1, zero_phase=True))
                Fs = Fs/dec
                LFP = np.array(fillLFP)
                fillLFP = []
                for i in range(len(LFPresampled)):
                    fillLFP.append(sc.signal.decimate(LFPresampled[i], dec, n=None, ftype='iir', axis=-1, zero_phase=True))
                LFPresampled = np.array(fillLFP)
            
            #slicing data into windows
            for i in range(int(np.floor(len(LFP[0])/(totdur*Fs)))) :
                tolook1 = LFP[:, int(totdur*Fs*i):int(totdur*Fs*i + totdur*Fs)]
                tolook2 = LFPresampled[:, int(totdur*Fs*i):int(totdur*Fs*i + totdur*Fs)]
                
                #power spectrum
                for electrode in range(len(tolook1)):  
                    freq, pwrsp = powerspeccalc(pwrspwin*Fs, tolook1[electrode], Fs)
                    pwrspdata.append([date, subtime, electrode + 1, totdur, i, pwrspwin, Fs, freq, pwrsp, False])
                    # date, subtime, electrode, total duration of powerspec analyzed here, total duration window, window over which power spec is calculated
                    # sampling frequency, frequency array of power spectrum, power spectra for each pwrpswin within this totdur window, is it resampled?
                
                #power spectrum
                for electrode in range(len(tolook2)):  
                    freq, pwrsp = powerspeccalc(pwrspwin*Fs, tolook2[electrode], Fs)
                    pwrspdata.append([date, subtime, electrode, totdur, i, pwrspwin, Fs, freq, pwrsp, True])
                
                #PAC
                if i == 1:
                    for electrode in range(len(tolook1)):
                        for pacwin in range(int(np.floor(len(tolook1[0])/(pacwindow*Fs)))):
                            windowdata = tolook1[electrode,int(pacwin*(pacwindow*Fs)): int(pacwin*(pacwindow*Fs) + (pacwindow*Fs))]
                            
                            p = Pac(f_pha=f_pha, f_amp=f_amp, dcomplex='hilbert', idpac = (2, 3, 0))
                            xpac = p.filterfit(Fs, windowdata, n_jobs=n_jobs).squeeze()
                            pval = p.infer_pvalues(p=0.05)        
                            xpac_smean = xpac[pval < .05].mean()  
                            pacdata.append([date, subtime, electrode + 1, totdur, i,pacwindow,pacwin, Fs, 'tort', False,  p, xpac, pval, xpac_smean])
                            # date, subtime, electrode, total duration analyzed, total duration window, window over which pac is calculated
                            # sampling frequency,is it resampled?, pacmethod, pacresults, pacresults2, where pvals are significant, mean pval

                        
                            #p = Pac(f_pha=f_pha, f_amp=f_amp, dcomplex='hilbert', idpac = (6, 3, 0))
                            #xpac = p.filterfit(Fs, windowdata, n_jobs=n_jobs).squeeze()
                            #pval = p.infer_pvalues(p=0.05)        
                            #xpac_smean = xpac[pval < .05].mean()  
                            
                            #pacdata.append([date, subtime, electrode, totdur, i,pacwindow, pacwin, Fs, 'gc', False,  p, xpac, pval, xpac_smean])
                            
                    for electrode in range(len(tolook2)):
                        for pacwin in range(int(np.floor(len(tolook2[0])/(pacwindow*Fs)))):
                            windowdata = tolook2[electrode,int(pacwin*(pacwindow*Fs)): int(pacwin*(pacwindow*Fs) + (pacwindow*Fs))]
                            
                            p = Pac(f_pha=f_pha, f_amp=f_amp, dcomplex='hilbert', idpac = (2, 3, 0))
                            xpac = p.filterfit(Fs, windowdata, n_jobs=n_jobs).squeeze()
                            pval = p.infer_pvalues(p=0.05)        
                            xpac_smean = xpac[pval < .05].mean()  
                            pacdata.append([date, subtime, electrode, totdur, i,pacwindow, pacwin, Fs, 'tort', True,  p, xpac, pval, xpac_smean])
                            # date, subtime, electrode, total duration analyzed, total duration window, window over which pac is calculated
                            # sampling frequency,is it resampled?, pacmethod, pacresults, pacresults2, where pvals are significant, mean pval

                        
                            #p = Pac(f_pha=f_pha, f_amp=f_amp, dcomplex='hilbert', idpac = (6, 3, 0))
                            #xpac = p.filterfit(Fs, windowdata, n_jobs=n_jobs).squeeze()
                            #pval = p.infer_pvalues(p=0.05)        
                            #xpac_smean = xpac[pval < .05].mean()  
                            
                            #pacdata.append([date, subtime, electrode, totdur, i, pacwindow, pacwin, Fs, 'gc', True,  p, xpac, pval, xpac_smean])
                    
            
#saving the outputted data in my own folders            
    with open("/scratch/achanna/kylie_edits/" + date + "_pwrsp_dec.pickle", "wb") as fp:   #Pickling
        pickle.dump(pwrspdata, fp)
        
    with open("/scratch/achanna/kylie_edits/" + date + "_pac_dec.pickle", "wb") as fp:   #Pickling
        pickle.dump(pacdata, fp)
            
            

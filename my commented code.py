import pandas as pd
import numpy as np
import math
import pickle
import glob
import tdt
import h5py
from scipy.fft import fft as fft 
from tensorpac import Pac
from tensorpac.signals import pac_signals_tort
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy as sc

def nextpow2(x):
    res = np.ceil(np.log2(x))   #finding exponent n for 2^n that leads to value ≥ x
                                #this is to set up for the fact that FFTs are fastest at lengths that are powers of two (eg. 512, 1024, etc)
    return res.astype('int')    #converting result from ceil to integer instead of float, for future calculations  



def ephysanalysis(window, ephysdata, samplefreq):   #'window' represents the number of voltage points in each window
                                                    #'sampling frequency' represents number of points per second
    npoints = len(ephysdata)                #number of total voltage points in the ephysdata
    nsegment = int(round(npoints/window)-1) #number of total windows in the ephysdata
    NFFT = 2**(nextpow2(npoints/nsegment))  #number of voltage points per window, rounded UP to nearest power of two. Then 2 is raised to this value
                                            #NFFT value represents figuring out how many points you want to process for your FFT calculation
                                            #the "fast" in FFT means that it takes an array of eg. 8 samples, and splits it in half, then processing it as two groups at same time instead of one at a time (and it does this over & over again, combining them at the end)
    spec = np.zeros([nsegment, NFFT])       #creating EMPTY 2-D array of zeros to hold all FFT results in future
                                            #has window numbers (eg. Window 1,2, etc.) along vertical axis and specific frequency values along horizontal axis (eg. f = 1 Hz, 2 Hz, etc.)
                                            #each cell in the array/table represents the power of a specific freqnecy in a specific window
    
    for ii in range(nsegment):                                      #going through each window                                              
        tempval = ii*window                                         #starting sample of each window (eg. 3 * 3000 = 9000, representing the start of the 9000-11999 window)
        fftcalc = ephysdata[int(tempval):int(tempval+window-1)]     #extract this specific window only (eg. 9000-11999)
        #Hanning window to reduce edging effects
        fftcalc = np.multiply(np.hanning(len(fftcalc)),fftcalc-np.mean(fftcalc))    #DC offset is the average/midvalue of your signal - ideal is 0, but offset leads to higher/lower (eg. +0.2 V, -0.3 V which is more real since electrodes can sit in tissue w/ small potentials)
                                                                                    #need to subtract this offset to ensure we get a non-zero Hz value from FFT calculation (if we have a non-zero offset, we get a large 0 Hz spike which we don't want)
                                                                                    #overall purpose of this line is to remove offset and smoothen out curve instead of discrete points/jumps in time
        spec[ii] = fft(fftcalc,NFFT)/npoints                                        #Compute FFT and putting each into the array from earlier, also normalizing each computed value
                                                                                    #each FFT calculation on a window yields a whole set of values: frequencies and powers/amplitudes
    freq = samplefreq/2*np.linspace(0,1,int(NFFT/2))                                #1D array that holds all the x-axis labels (the frequency values of all waves present)
    pwrsp = 2 * abs(spec[:,0: int(len(spec[0])/2)])                                 #these are the power values that fill in the table for all windows
    return freq, pwrsp

pwrspwindow = 3                                                                         #3 second window to computer power spec (only rlly need this long to understand what frequencies are present in a window)
pacwindow = 10                                                                          #10 second window to compute modulation index (need longer window since PAC measures relationships - ie. between slow phase and fast amplitude)

#Phase-Amplitude Coupling (Try every low & every high-frequency, and test whether the phase of the slow one modulates the amplitude of the fast one)
f_amp = (30, 140, 5, 1)  #amplitude frequencies for MI calc (start, end, stepsize)      #specifying we only care about the gamme/HIGH-frequency bands' power (ie. wave frequencies between 30-140 Hz, each filtered w/ a 1 Hz bandwidth)
f_pha = (1, 29, 2, .5) #phase frequencies for MI calc (start, end, stepsize)            #specifying we only care about the LOW-frequency bands' phase (ie. wave frequencies between 1-29 Hz, each filtered w/ a 0.5 Hz bandwidth)
                                                                                        #testing to see if relationship between these two
                                                                                        #"Modulation Index" measures how strongly the amplitude of a high-frequency oscillation (eg. 80 Hz gamma) depends on the phase of a slow-frequency oscillation (eg. 6 Hz theta)
                                                                                        #
n_perm = 200                                                                            #number of permutations for modulation index calculation
                                                                                        #essentially calculates MI for you multiple times based on randomizing the values, and allows you to know if your real MI is correct or influenced by noise etc. (w/ statistical significance p<0.05)
n_jobs = 18                                                                             #number of cpus to run (more CPUs = faster)
dec = 10                                                                                #if decimating LFP signal, keeps only every 10th data point so you can process values faster (shrinks the dataset by 10x)

dir = '/directorylocation' #edit to include directory location
data = tdt.read_block(dir)                                                              #opens TDT recording
LFP = data.streams.LFP1.data                                                            #refers to extracting the actual LFP voltage recordings from all electrodes
                                                                                        #LFP1 = a specific stream (the first LFP channel group)
Fs = data.streams.LFP1.fs                                                               #Extract & store the sampling rate (fs) of this recording
pwrspwin = pwrspwindow*Fs                                                               #number of points per window /// per 3-second chunk we specified (ie. comes from multiplying seconds per window * samples per second)
                                                                                        #important because every FFT needs to know exactly how many points to take for each window
#this is all optional    
fillLFP = []
if dec > 1:    
    for i in range(len(LFP)):
        fillLFP.append(sc.signal.decimate(LFP[i], dec, n=None, ftype='iir', axis=-1, zero_phase=True))
    Fs = Fs/dec
LFP = np.array(fillLFP)
print(LFP.shape)

    
pwrspwin = pwrspwindow*Fs   #repeat line 
pwrspdata = []              #empty list where results will be added to once calculated
pacdata = []                #empty list


for elec in range(16): #16 electrode channels
    tolook = LFP[elec]
    freq, pwrsp = ephysanalysis(pwrspwin, tolook, Fs)
    pwrspdata.append([elec, freq, pwrsp, pwrspwin])                                                     #each electrode's power-spectrum analysis will be added to this list, for a total of 16 entries, each with the following info:
                                                                                                        #elec = electrode number
                                                                                                        #freq = 

    for pacwin in range(int(np.floor(len(tolook)/(pacwindow*Fs)))):
        windowdata = tolook[int(pacwin*(pacwindow*Fs)): int(pacwin*(pacwindow*Fs) + (pacwindow*Fs))]
        p = Pac(f_pha=f_pha, f_amp=f_amp, dcomplex='hilbert', idpac = (2, 3, 0))
        xpac = p.filterfit(Fs, windowdata, n_jobs=n_jobs).squeeze()
        pval = p.infer_pvalues(p=0.05) 
        xpac_smean = xpac[pval < .05].mean()
        pacdata.append([elec, pacwin, pacwindow, p, xpac, pval, xpac_smean])
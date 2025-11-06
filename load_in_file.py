# import pandas as pd
import numpy as np
# import math
# import pickle
# import glob
# import tdt
# import h5py
from scipy.fft import fft as fft
from tensorpac import Pac
from tensorpac.signals import pac_signals_tort
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy as sc

from generatingsignal import y_high, y_low, time, sf


def nextpow2(x):
    res = np.ceil(np.log2(x))
    return res.astype('int')  # we want integer values only but ceil gives float


# generate power spectra
def ephysanalysis(window, ephysdata, samplefreq):  # enter ephys analysis window in samples, full data to analyze, sampling frequency of data
    npoints = len(ephysdata)
    nsegment = int(round(npoints / window) - 1)
    NFFT = 2 ** (nextpow2(npoints / nsegment))
    spec = np.zeros([nsegment, NFFT])

    for ii in range(nsegment):
        # Split up section into subsections
        tempval = ii * window
        fftcalc = ephysdata[int(tempval):int(tempval + window - 1)]
        # Hanning window to reduce edging effects
        fftcalc = np.multiply(np.hanning(len(fftcalc)), fftcalc - np.mean(fftcalc))
        # Compute FFT at NFFT resolution
        spec[ii] = fft(fftcalc, NFFT) / npoints

    freq = samplefreq / 2 * np.linspace(0, 1, int(NFFT / 2))
    pwrsp = 2 * abs(spec[:, 0: int(len(spec[0]) / 2)])
    return freq, pwrsp


pwrspwindow = 3  # 3 second window to computer power spec
pacwindow = 10   # 10 second window to compute modulation index

f_amp = (30, 140, 5, 1)   # amplitude frequencies for MI calc (start, end, stepsize)
f_pha = (1, 29, 2, .5)    # phase frequencies for MI calc (start, end, stepsize)
n_perm = 200               # number of permutations for modulation index calculation
n_jobs = 18                # number of cpus to run
# dec = 10  # if decimating LFP signal

# dir = '/directorylocation'  # edit to include directory location
# data = tdt.read_block(dir)
# LFP = data.streams.LFP1.data
# Fs = data.streams.LFP1.fs

LFP = np.vstack([y_high, y_low])  # this is new
Fs = int(sf)                      # this is new
pwrspwin = pwrspwindow * Fs

"""
fillLFP = []
if dec > 1:
    for i in range(len(LFP)):
        fillLFP.append(sc.signal.decimate(LFP[i], dec, n=None, ftype='iir', axis=-1, zero_phase=True))
    Fs = Fs / dec
LFP = np.array(fillLFP)
print(LFP.shape)
"""

pwrspwin = pwrspwindow * Fs

pwrspdata = []
pacdata = []

for elec in range(LFP.shape[0]):  # this is new
    tolook = LFP[elec]
    freq, pwrsp = ephysanalysis(pwrspwin, tolook, Fs)
    pwrspdata.append([elec, freq, pwrsp, pwrspwin])

    for pacwin in range(int(np.floor(len(tolook) / (pacwindow * Fs)))):
        windowdata = tolook[int(pacwin * (pacwindow * Fs)): int(pacwin * (pacwindow * Fs) + (pacwindow * Fs))]
        p = Pac(f_pha=f_pha, f_amp=f_amp, dcomplex='hilbert', idpac=(2, 3, 0))
        xpac = p.filterfit(Fs, windowdata, n_jobs=n_jobs).squeeze()
        pval = p.infer_pvalues(p=0.05)
        xpac_smean = xpac[pval < .05].mean()
        pacdata.append([elec, pacwin, pacwindow, p, xpac, pval, xpac_smean])

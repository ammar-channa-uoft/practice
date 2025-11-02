
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorpac.signals import pac_signals_tort

sf = 1024
duration_sec = 120                 # 2 minutes = 120 seconds
n_times = int(sf * duration_sec)   # number of points for each signal = sampling frequency * seconds = 1024 * 120 = 122880
n_epochs = 1
f_pha = 10.0
f_amp = 100.0
noise = 1.0
rnd_state = 0
chi_high = 0.05            # high coupling case
chi_low  = 0.95            # low coupling case

short = int(0.5 * sf)


# Generate signals
signal_high, time = pac_signals_tort(f_pha=f_pha, f_amp=f_amp, sf=sf, n_times=n_times, n_epochs=n_epochs,
    chi=chi_high, noise=noise, dpha=0.0, damp=0.0, rnd_state=rnd_state)

signal_low, time = pac_signals_tort(f_pha=f_pha, f_amp=f_amp, sf=sf, n_times=n_times, n_epochs=n_epochs,
    chi=chi_low, noise=noise, dpha=0.0, damp=0.0, rnd_state=rnd_state)

y_high = signal_high[0,0:n_times]
y_low  = signal_low[0,0:n_times]

y_high_short = y_high[:short]
y_low_short = y_low[:short]

time_short = time[:short]

# Plot
plt.figure(0)
plt.plot(time_short, y_high_short)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("HIGH Coupling Signal (chi=0.05)")

plt.savefig("high_coupling.png")
plt.close()

plt.figure(1)
plt.plot(time_short, y_low_short)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("LOW Coupling Signal (chi=0.95)")

plt.savefig("low_coupling.png")
plt.close()


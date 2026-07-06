import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.signal as signal
import torch.nn as nn
from neurostreak.model import BackScatter, Water, AquisticGeometry
import scipy.fft as sp_fft
template = np.load('data/clean_water_10m/template.npy')


def stft( x):
    fs = 68.27 * 10**3
    wind = 256
    noverlap = wind // 2

    spectrograms = []
    freqs = None
    times = None

    if x.ndim == 1:
        x = x[np.newaxis, :]
    original_shape = x.shape
    flattened_x = x.reshape(-1, x.shape[-1])

    for row in flattened_x:
        row = row - np.median(row)
        row = np.pad(row, (0, 65535 - len(row)), mode='constant', constant_values=0)
        f, t, Zxx = signal.stft(row, fs=fs, nperseg=wind, noverlap=noverlap)
        if freqs is None:
            freqs, times = f, t
        spectrograms.append(np.abs(Zxx))
    spectrograms = np.array(spectrograms)
    new_shape = original_shape[:-1] + (spectrograms.shape[1], spectrograms.shape[2])
    result = spectrograms.reshape(new_shape)

    return result, freqs, times

result, freqs, times = stft(template)
print(result.shape)
print(freqs.shape)
print(np.abs(result))
print(np.angle(result))
print()
print(freqs)

print(result[0])

plt.imshow(result[0][:100, :1000])
plt.show()
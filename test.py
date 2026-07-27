import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.signal as signal
import torch.nn as nn
import scipy.fft as sp_fft
import cv2
from types import SimpleNamespace
from PIL import Image
from fontTools.config import Config

import torch
from dataclasses import dataclass

from neurostreak.noise import CEQPF
from neurostreak.noise import CEAPF

import pywt




template = np.load('data/clean_water_10m/template.npy')
# template = torch.Tensor(template)

lst_template = [template for _ in range(2048)]
lst_template = np.array(lst_template)
img = np.array(Image.open('data/clean_water_10m/data/069.tif'))



@dataclass
class Config:
    absorption: float
    scattering: float
    indicatrix: float
    C1: float
    C2: float
    alpha: float
    beta: float
    light_speed: float
    FOV: int

config = Config(
    absorption= 0.179,
    scattering= 0.289,
    indicatrix=0.9,
    C1= 4.888e-7,
    C2= 0.4179,
    alpha=  -0.03681,
    beta = 3.019,
    light_speed = 2.237e8,
    FOV = 180,
)
#
#
# water = Water(
#     absorption = config.absorption,
#     scattering = config.scattering,
#     indicatrix = config.indicatrix,
# )
#
# ceapf = CEAPF(
#     absorption = config.absorption,
#     C1 = config.C1,
#     C2 = config.C2,
#     alpha = config.alpha,
#     beta = config.beta,
#     light_speed = config.light_speed,
#     FOV = config.FOV,
#
# )
#
# filter = ceapf.simulate(40, template)
# k= 505
# fig, ax = plt.subplots(4,1)
# ax[0].plot(filter[0])
# ax[1].plot(template)
# ax[2].plot(img[1000][k:k+len(template)] )
# ax[3].plot(img[1000][k:k+len(template)] /img[1000][k:k+len(template)].sum()  - template/template.sum() )
# ax[3].plot(img[1000][k:k+len(template)]  /img[1000][k:k+len(template)].sum()  - filter[0][0:len(template)] / filter[0].sum() )
# plt.show()



template = template - np.median(template, axis=0)
template = np.pad(template, (0, 2048 - len(template)), mode='constant',constant_values=0)

lst_template = np.array([template for _ in range(2048)])

img = img - np.median(img, axis=1)
print(lst_template.shape)
print(img.shape)

from neurostreak.nn_layers import WaveletAmplitudeEmbeddingBlock
from neurostreak.model import EmbeddingNeuroStreak

emb = EmbeddingNeuroStreak(max_amp=50, step = 0.5)
spectral_signal, spectral_template, wavelet_signal, wavelet_template = emb(template = lst_template, signal = img)

print(spectral_signal.shape)
print(spectral_template.shape)
print(wavelet_signal.shape)
print(wavelet_template.shape)

plt.imshow(wavelet_signal[1024])
plt.show()
plt.imshow(wavelet_template[1024])
plt.show()


fig, ax = plt.subplots(2,1)
ax[0].plot(lst_template[1024])
ax[1].plot(img[1024])
plt.show()

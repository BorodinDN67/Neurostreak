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


import pywt




template = np.load('data/clean_water_10m/template.npy')
# template = torch.Tensor(template)

lst_template = [template for _ in range(2048)]
lst_template = np.array(lst_template)
img = np.array(Image.open('data/clean_water_10m/data/069.tif'))[:32]



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





template = template - np.median(template, axis=0)
template = np.pad(template, (0, 2048 - len(template)), mode='constant',constant_values=0)

lst_template = np.array([template for _ in range(32)])

img = img - np.median(img, axis=1, keepdims=True)
print(lst_template.shape)
print(img.shape)

from neurostreak.nn_layers import WaveletAmplitudeEmbeddingBlock
from neurostreak.model import EmbeddingNeuroStreak
from neurostreak.model import NeuroStreakBackbone

emb = EmbeddingNeuroStreak(max_amp=50, step = 0.5)
bkb = NeuroStreakBackbone(
    hidden_2d_dim_1=64,
    hidden_2d_dim_2=128,
    kernel_size=(3,3),
    padding=(1,1),
    num_heads=8,
    embed_dim=512,
    depth=8,
    max_pool_kernel_size_1=(2,2),
    max_pool_stride_1=(2,2),
    max_pool_kernel_size_2=(2,2),
    max_pool_stride_2=(2,2),
    max_pool_kernel_size_3=(2,2),
    max_pool_stride_3=(2,2),
    num_scales = 64
)
spectral_signal, spectral_template, wavelet_signal, wavelet_template = emb(template = lst_template, signal = img)
template_wvlt_bkbn, signal_wvlt_bkbn, template_spctrl_bkbn, signal_spctrl_bkbn = (
    bkb(
    spectral_signal,
    spectral_template,
    wavelet_signal,
    wavelet_template
))

print(template_wvlt_bkbn.shape)  #torch.Size([32, 384, 512])
print(signal_wvlt_bkbn.shape) #torch.Size([32, 384, 512])
print(template_spctrl_bkbn.shape) #torch.Size([32, 1, 512])
print(signal_spctrl_bkbn.shape) #torch.Size([32, 512, 8])


# print(spectral_signal.shape)  #torch.Size([2048, 1, 512])
# print(spectral_template.shape) #torch.Size([2048, 1, 512])
# print(wavelet_signal.shape) #torch.Size([2048, 98, 2048])
# print(wavelet_template.shape) #torch.Size([2048, 98, 2048])
#
# plt.imshow(wavelet_signal[1024])
# plt.show()
# plt.imshow(wavelet_template[1024])
# plt.show()
#
#
# fig, ax = plt.subplots(2,1)
# ax[0].plot(lst_template[1024])
# ax[1].plot(img[1024])
# plt.show()

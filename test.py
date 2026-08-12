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
from torch.xpu import device

template = np.load('data/clean_water_20m/template.npy')
# template = torch.Tensor(template)

lst_template = [template for _ in range(2048)]
lst_template = np.array(lst_template)
img = np.array(Image.open('data/clean_water_20m/data/069.tif'))



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
from neurostreak.model import NeuroStreakEmbedding
from neurostreak.model import NeuroStreakBackbone
from neurostreak.model import NeuroStreakHead
from src.config.config import Config
from pathlib import Path
import ssqueezepy as sq
import seaborn as sns
import ptwt

test_template = torch.tensor(lst_template[0])
test_img = img[100]

lenght = test_template.shape[-1]

signal_3 = torch.cat([test_template, test_template, test_template], dim=-1)
coef, freqs = ptwt.cwt(signal_3,np.arange(1,101, 1), wavelet='gaus2')
print(coef.shape)
coef = coef[:, test_template.shape[-1] : 2 * test_template.shape[-1] ]
print(coef.shape)

# coef = coef.permute(1, 0, 2)

print(coef.shape)
plt.imshow(coef)
plt.plot(lst_template[0])
plt.show()

# Wx, scales = sq.cwt(test_template, 'morlet', fs = 102.4 * 10**9 )
# amplitude = np.abs(Wx)
# phase = np.angle(Wx)
# sq.visuals.imshow(amplitude,yticks=scales, cmap='gray', )
# sq.visuals.imshow(phase,yticks=scales, cmap='gray', )
#
# Wx, scales = sq.cwt(test_template, 'hhhat', fs = 102.4 * 10**9 )
# amplitude = np.abs(Wx)
# phase = np.angle(Wx)
# sq.visuals.imshow(amplitude,yticks=scales, cmap='gray', )
# sq.visuals.imshow(phase,yticks=scales, cmap='gray')
#
# Wx, scales = sq.cwt(test_template, 'cmhat', fs = 102.4 * 10**9 )
# amplitude = np.abs(Wx)
# phase = np.angle(Wx)
# sq.visuals.imshow(amplitude,yticks=scales, cmap='gray')
# sq.visuals.imshow(phase,yticks=scales, cmap='gray')
#
# Wx, scales = sq.cwt(test_template, 'bump', fs = 102.4 * 10**9 )
# amplitude = np.abs(Wx)
# phase = np.angle(Wx)
# sq.visuals.imshow(amplitude,yticks=scales, cmap='gray')
# sq.visuals.imshow(phase,yticks=scales, cmap='gray')

# config_path = Path('src/config/config.yaml')
# config = Config()
# config.load_model_config(config_path)
# model = config.get_model_from_config()
# print(model)


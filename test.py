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

from neurostreak.model import Embedding
import torch
from dataclasses import dataclass

from neurostreak.model.embedding import Water, MTF_Dolin
from neurostreak.noise import CEQPF
from neurostreak.noise import CEAPF

import pywt




template = np.load('data/clean_water_20m/template.npy')
# template = torch.Tensor(template)

lst_template = [template for _ in range(2048)]
lst_template = np.array(lst_template)
img = np.array(Image.open('data/clean_water_20m/data/002.tif'))



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



template = template - np.mean(template, axis=0)
template = np.array([template])
print(template.shape)

from neurostreak.model.embedding import WaveletAmplitudeEmbeddingBlock

emb = WaveletAmplitudeEmbeddingBlock(max_amp= 100, step=1,wavelet='gaus1')
res = emb(template)
res_img = emb(img)
print(res.shape)
plt.imshow(
    res[0],
    aspect='auto',
    cmap='jet',
    origin='lower'

)
plt.show()
print(res_img.shape)
plt.imshow(
    res_img[1000],
    aspect='auto',
    cmap='jet',
    origin='lower'

)
plt.show()
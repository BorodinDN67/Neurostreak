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


template = np.load('data/clean_water_13m/template.npy')
template = torch.Tensor(template)

lst_template = [template for _ in range(2048)]
lst_template = np.array(lst_template)
img = torch.Tensor(np.array(Image.open('data/clean_water_13m/data/067.tif')))



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
    absorption= 0.3666,
    scattering= 1.289,
    indicatrix=0.9,
    C1= 1.677e-6,
    C2= 0.2730,
    alpha=  0.6577,
    beta = 3.169,
    light_speed = 2.237e8,
    FOV = 20,
)

# emb = Embedding(config = config)
# res = emb.forward(lst_template, img)
# print(res)
#
# spectral, matching = res
# # print(spectral[0])
# print(spectral[0].shape)
# print('////////////////////////////////////////////////////////////////////////////////////////////////')
# # print(spectral[1])
# print(spectral[1].shape)
# print('////////////////////////////////////////////////////////////////////////////////////////////////')
# # print(matching)
# print(matching.shape)
# print('////////////////////////////////////////////////////////////////////////////////////////////////')
# print(matching[1000])
# plt.plot(matching[1000,:])
# plt.show()
#
#
water = Water(
    absorption = config.absorption,
    scattering = config.scattering,
    indicatrix = config.indicatrix,
)

ceapf = CEAPF(
    absorption = config.absorption,
    C1 = config.C1,
    C2 = config.C2,
    alpha = config.alpha,
    beta = config.beta,
    light_speed = config.light_speed,
    FOV = config.FOV,

)

filter = ceapf.simulate(40, template)

fig, ax = plt.subplots(3,1)
ax[0].plot(filter[0])
ax[1].plot(template)
ax[2].plot(img[1000][500:1500])
plt.show()

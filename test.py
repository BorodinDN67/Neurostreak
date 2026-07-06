import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.signal as signal
import torch.nn as nn
from neurostreak.model import BackScatter, Water, AquisticGeometry
import scipy.fft as sp_fft
from neurostreak.noise import *


template = np.load('data/clean_water_10m/template.npy')
tail = template[500:]
template  = template -  np.mean(tail)

water = Water(
    absorption = 0.2,
    scattering = 0.2,
    indicatrix = 0.95
)

model = MTF_Dolin(
    water = water,
    k_theta = 50,
    L = 128,
    mu = 0.15,
    theta0 = 1.0,
    gamma = 0.4,
    k = 2.0,
)
im_1 = [template for i in range(len(template))]
im_2 = [model.simulate(template = template, distance = 13) for i in range(len(template))]

fig, ax = plt.subplots(3,2)
ax[0,0].imshow(im_1)
ax[0,1].imshow(im_2)
ax[1,0].plot(template)
ax[1,1].plot(model.simulate(template = template, distance = 13))
ax[2,0].plot(template / max(template))
ax[2,0].plot(model.simulate(template = template, distance = 13) / max(model.simulate(template = template, distance = 13)))
plt.show()


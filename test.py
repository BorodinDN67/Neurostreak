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
from neurostreak.model import NeuroStreakEmbedding
from neurostreak.model import NeuroStreakBackbone
from neurostreak.model import NeuroStreakHead
from src.config.config import Config
from pathlib import Path


config_path = Path('src/config/config.yaml')
config = Config()
config.load_model_config(config_path)
model = config.get_model_from_config()
print(model)


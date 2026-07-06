import torch.nn as nn
import  numpy as np
from scipy.signal import fftconvolve, correlate
from scipy.special import gamma
from sympy.codegen.ast import none
from sympy.physics.units import velocity


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        pass
    def forward(self, x):
        pass



def get_matched_filters(template):
    templates = []


    return templates


def matched_filters(x, template):
    filters = get_matched_filters(template)
    pass





#генеральные параметры воды
class Water():
    def __init__(self,absorption = None,scattering = None,anisotropy = None, refractive_index = None, theta_scale = None ):
        self.absorption = absorption #поглощение
        self.scattering = scattering #рассеяние
        self.anisotropy = anisotropy #Анизотропия
        self.refractive_index = refractive_index #переотражение

        self.attenuation = self.absorption + self.scattering
        self.theta_scale = theta_scale


#Временные, пространственные искажениям
class AquisticGeometry():
    def __init__(self,water ):
        self.water = water
        self.c0 = 299792458.0
        self.balistic_fotons = 0.8

    def simulate(self,template,type = 'exponential'):
        if type == 'exponential':
            time = np.arange(len(template), dtype = np.float64)
            time *= 30 * 10** -9 / len(template)

            velocity = self.c0 / self.water.refractive_index

            simulate_signal = np.exp( -self.water.attenuation * velocity * time )

            degraded_signal = template * simulate_signal
            return degraded_signal

#Обратное рассеяние
class BackScatter():
    def __init__(self, water,scattered_fraction, shape_gamma = 2, theta_scale = 80*10**-12 ):
        self.water = water
        self.shape_gamma = shape_gamma
        self.theta_scale = theta_scale
        self.scattered_fraction = scattered_fraction

    def kernel_gamma(self, template):
        t = np.arange(len(template), dtype = np.float64) * 30 / 840 * 10 ** -9
        scattered = t ** (self.shape_gamma - 1) * np.exp( -t / self.theta_scale  ) /( gamma(self.shape_gamma) * self.theta_scale ** self.shape_gamma)
        scattered *= 30 / 840 * 10 ** -9
        # scattered /= np.max(scattered.sum(), 1e-12)

        kernel = self.scattered_fraction * scattered
        kernel[0] = 1.0 - self.scattered_fraction
        kernel /= np.max(kernel)

        return kernel


    def simulate(self, template):
        kernel_gamma = self.kernel_gamma(template)
        backscatter = fftconvolve(template, kernel_gamma, mode = 'full')[:len(template)]
        return backscatter
#Итог

class ModelWater():
    pass
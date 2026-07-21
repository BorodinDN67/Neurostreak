import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class CEAPF:
    def __init__(
            self,
            C1: float,
            C2: float,
            alpha: float,
            beta: float,
            light_speed: float = 2.237e8,
            absorption: float = 0.333,
            scattering: float = 1.688,
            FOV: int = 20,
            dt: float = None,
            kernel_size: int = 24
    ):
        super(CEAPF, self).__init__()

        self.C1 = C1
        self.C2 = C2
        self.alpha = alpha
        self.beta = beta
        self.light_speed = light_speed
        self.absorption = absorption
        self.FOV = FOV
        self.dt = 30/2048 *1e-9 if dt is None else dt
        self.kernel_size = kernel_size

    def make_kernal(self, distance):

        dt_arr = torch.arange(1, self.kernel_size+1) * self.dt
        t0 = distance / self.light_speed

        h = self.C1 * dt_arr.pow(self.alpha) * torch.exp(-1*self.absorption * self.light_speed * (dt_arr + t0)) / (dt_arr + self.C2).pow(self.beta)
        kernel = h * np.random.exponential(size=h.shape, scale=10.1)
        kernel = kernel / sum(kernel)

        return kernel

    def simulate(self, distance, template):
        kernel = self.make_kernal(distance)

        tmp = template.unsqueeze(0).unsqueeze(0)
        kernel = kernel.to(dtype=torch.float32)
        krnl = kernel.flip(0).unsqueeze(0).unsqueeze(0)

        y_batch = F.conv1d(tmp, krnl, padding=krnl.numel()//2)

        return y_batch.squeeze(0)

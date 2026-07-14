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
            FOV: int = 20
    ):
        super(CEAPF, self).__init__()

        self.C1 = C1
        self.C2 = C2
        self.alpha = alpha
        self.beta = beta
        self.light_speed = light_speed
        self.absorption = absorption
        self.FOV = FOV

    def make_kernal(self, distance):
        dt = 0.1e-9
        dt_arr = torch.arange(0, 3 * distance/self.light_speed , dt)
        t0 = distance / self.light_speed

        dt_arr = dt_arr - t0
        h = self.C1 * (dt_arr ** self.alpha) / ((dt_arr + self.C2)**self.beta) * torch.exp(-self.absorption * self.light_speed * (dt_arr + t0))
        h[dt_arr < 0] = 0
        kernel = h / max(h)
        return kernel

    def simulate(self, distance, template):
        kernel = self.make_kernal(distance)

        tmp = template.unsqueeze(0).unsqueeze(0)
        krnl = kernel.unsqueeze(0).unsqueeze(0)

        y_batch = F.conv1d(tmp, krnl, padding='same')

        return y_batch.squeeze(0)

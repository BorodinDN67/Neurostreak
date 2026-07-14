import torch
import torch.nn as nn
import torch.nn.functional as F

class CEQPF(nn.Module):
    def __init__(
            self,
            c1: float,
            c2_ns: float,
            alpha: float,
            beta: float,
            water,
            c: float = 2.237e8,
            sample_period_ns: float = 2.5,
            kernel_size: int = 128,
            eps: float = 1e-12
    ):
        super(CEQPF, self).__init__()

        self.c1 = c1
        self.c2_ns = c2_ns
        self.alpha = alpha
        self.beta = beta
        self.absorption = water.absorption
        self.c = c
        self.sample_period_ns = sample_period_ns
        self.kernel_size = kernel_size
        self.eps = eps

    def make_kernel(self,distance: float, device: torch.device, dtype: torch.dtype):
        path_length = 2.0 * distance

        dt_ns = torch.arange(
            self.kernel_size,
            device = device,
            dtype = dtype
        ) * self.sample_period_ns

        dt_s = dt_ns * 1e-9
        dt_safe_ns = dt_ns.clamp_min(self.eps)
        scattering_shape = dt_safe_ns.pow(self.alpha) / (dt_safe_ns + self.c2_ns).pow(self.beta)

        absorption_term = torch.exp( -self.absorption * (path_length + self.c * dt_s))

        kernel = self.c1 * scattering_shape * absorption_term
        kernel = kernel / kernel.sum().clamp_min(self.eps)

        return kernel

    def simulate(
            self,
            distance: float,
            template: torch.Tensor,
    ):
        kernel = self.make_kernel(distance, device = template.device, dtype = template.dtype)
        weight = kernel.flip(0)[None,None,:]

        x_padded = F.pad(template, (self.kernel_size-1, 0))
        template_padded = x_padded.flip(0)[None,None,:]
        distored = F.conv1d(template_padded, weight).squeeze(0)

        return distored.flip(dims=[-1])



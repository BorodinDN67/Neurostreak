import numpy as np
import scipy
import scipy.signal as signal_
import scipy.fft as sp_fft
import torch.nn as nn
import torch
from loguru import logger


from ..nn_layers import FDEmbeddingBlock, WaveletAmplitudeEmbeddingBlock





class NeuroStreakEmbedding(nn.Module):
    def __init__(
            self,
            wavelet: str = 'gaus1',
            max_amp: int = 100,
            step: float = 1,
            width: float =1.0,
            act: str ='silu'
        ):
        super().__init__()

        self.spectral_embedding_block = FDEmbeddingBlock(width=width, act=act )
        self.wavelet_amplitude_embedding_block = WaveletAmplitudeEmbeddingBlock(wavelet=wavelet, max_amp=max_amp, step=step)

    def forward(self, template, signal):
        spectral_signal = self.spectral_embedding_block(torch.Tensor(signal))
        spectral_template = self.spectral_embedding_block(torch.Tensor(template))

        wavelet_signal = self.wavelet_amplitude_embedding_block(signal)
        wavelet_template = self.wavelet_amplitude_embedding_block(template)

        return spectral_signal, spectral_template, wavelet_signal, wavelet_template



class StreakNetEmbedding(nn.Module):

    def __init__(self, config):
        super(StreakNetEmbedding, self).__init__()

        self.spectral_embedding_block = FDEmbeddingBlock()

    def forward(self, template, signal):
        logger.info(f'Начали получать эмбеддинг для батча: {signal.shape}')

        template_embedding = self.spectral_embedding_block(template)
        signal_embedding = self.spectral_embedding_block(signal)

        logger.success('Получили эмбеддинг')
        return  torch.concat([signal_embedding, template_embedding], dim=1)



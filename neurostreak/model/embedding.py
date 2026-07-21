import numpy as np
import scipy
import scipy.signal as signal_
import scipy.fft as sp_fft
import torch.nn as nn
import torch
from loguru import logger

from ..noise import Water, MTF_Dolin


class Embedding(nn.Module):

    def __init__(self, config):
        super(Embedding, self).__init__()
        self.config = config

        self.SpectralEmbeddingBlock = SpectralEmbeddingBlock()
        self.MatchigFiltersEmbedding = MatchigFiltersEmbedding(self.config)
        # self.DiffusionEmbedding = DiffusionEmbedding(self.signal)

    def forward(self, template, signal):
        logger.info(f'Начали получать эмбеддинг для батча: {signal.shape}')
        spectrogram = self.SpectralEmbeddingBlock(template=template, signal=signal)
        matching_filters = self.MatchigFiltersEmbedding(template=template, signal=signal)
        logger.success('Получили эмбеддинг')
        return [spectrogram, matching_filters]


#Блок, отвечающий за спектральный анализ входных данных.
class SpectralEmbeddingBlock(nn.Module):
    def __init__(self):
        super(SpectralEmbeddingBlock, self).__init__()

        self.fs = 68.27 * 10 ** 3

    def forward(self, signal, template):
        #todo: подрообнее разобраться с дополнением сигнала нулями - что это дает и зачем
        signal = torch.Tensor(signal)
        template = torch.Tensor(template)

        signal_freq = torch.fft.rfft(signal, 65536)
        signal_freq = signal_freq[:, :4000]
        signal_real = torch.real(signal_freq)
        signal_imag = torch.imag(signal_freq)

        template_freq = torch.fft.rfft(template, 65536)
        template_freq = template_freq[:, :4000]
        template_real = torch.real(template_freq)
        template_imag = torch.imag(template_freq)

        concat_signal = torch.concat([signal_real, signal_imag], dim=1)
        concat_template =  torch.concat([template_real, template_imag], dim=1)
        return [concat_signal, concat_template]




#блок эмбеддинга, отвечающий за соответсвие входящего сигнала и математической модели искажения сигнала
class MatchigFiltersEmbedding(nn.Module):

    def __init__(self, config):
        super(MatchigFiltersEmbedding, self).__init__()

        self.config = config

    def get_filters(self,template, config):
        filters = []

        water = Water(
        absorption = config.absorption,
        scattering = config.scattering,
        indicatrix = config.indicatrix
        )


        for distance in range(1,40):
            model = MTF_Dolin(
                water = water,
                k_theta = config.k_theta,
                k = config.k,
                L = config.L,
                mu = config.mu,
                theta0 = config.theta0,
                gamma = config.gamma,
            )
            filters.append(model.simulate(template, distance))

        return filters

    def forward(self, template, signal):
        batch_size, lenght = signal.shape
        template_norm = (template - np.median(template)) / np.std(template)
        all_filters = []
        res = [[] for _ in range(batch_size)]

        for b in range(batch_size):
            filters = self.get_filters(template_norm[b], self.config)
            all_filters.append(filters)

        num_filters = len(all_filters[0])

        for b in range(batch_size):
            signal_norm = (signal[b] - np.median(signal[b])) / np.std(signal[b])
            for f_idx, filter in enumerate(all_filters[b]):
                corr = signal_.correlate(signal_norm, filter, mode='valid')
                corr /= len(filter)

                res[b].append(np.max(corr))
        return torch.Tensor(scipy.special.softmax(res) )


class STTFEmbeddingBlock(nn.Module):
    def __init__(self):
        super(STTFEmbeddingBlock, self).__init__()

    def forward(self,x):
        wind = 256
        noverlap = wind // 2

        spectrograms = []
        freqs = None
        times = None

        if x.ndim == 1:
            x = x[np.newaxis, :]
        original_shape = x.shape
        flattened_x = x.reshape(-1, x.shape[-1])

        for row in flattened_x:
            row = row - np.median(row)
            row = np.pad(row, (0, 65535 - len(row)), mode='constant', constant_values=0)
            f, t, Zxx = signal_.stft(row, fs= self.fs, nperseg=wind, noverlap=noverlap)
            if freqs is None:
                freqs, times = f, t
            spectrograms.append(np.abs(Zxx))
        spectrograms = np.array(spectrograms)
        new_shape = original_shape[:-1] + (spectrograms.shape[1], spectrograms.shape[2])
        result = spectrograms.reshape(new_shape)

        return result
#Блок, очищающий входной снимок от лишних помех и позволяющий использовать исходный сигнал
# будет хорошо, если переедет в самое начала тракта очистки. Нужно тестить отдельно.


class WaveletAmplitudeEmbeddingBlock(nn.Module):
    def __init__(
        self,
        wavelet: str = 'gaus1',
        max_amp: int = 100,
        step: float = 1,
    ):
        super(WaveletAmplitudeEmbeddingBlock,self).__init__()
        self.max_amp = max_amp
        self.step = step
        self.wavelet = wavelet

    def forward(
        self,
        signal: np.ndarray,
    ):
        import pywt

        signal_3 = np.concatenate([signal,signal,signal], axis=-1)
        wdth = np.arange(1,self.max_amp, self.step)
        coef, freqs = pywt.cwt(signal_3, wdth, wavelet= self.wavelet)
        coef = coef[:,:,len(signal[0]):2 * len(signal[0]) ]
        coef = coef.transpose(1,0,2)
        return torch.Tensor(coef)

class DiffusionEmbedding(nn.Module):

    def __init__(self):
        super(DiffusionEmbedding, self).__init__()
        pass



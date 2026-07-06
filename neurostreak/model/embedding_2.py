import numpy as np
import scipy.stats as st
import scipy.signal as signal
import scipy.fft as sp_fft
import torch.nn as nn

#Блок, отвечающий за спектральный анализ входных данных.
class SpectralEmbeddingBlock(nn.Module):
    def __init__(self, template, signal):
        super(SpectralEmbeddingBlock, self).__init__()

        self.template = template # 1 x длина отсчетов
        self.signal = signal # высота щели х длина отсчетов
        self.fs = 68.27 * 10 ** 3

    def forward(self, x):

        fft_template = self.fft(self.template)
        fft_signal = self.fft(self.signal)

        stft_signal = self.stft(self.signal)
        stft_template = self.stft(self.template)

        return [fft_signal, fft_template, stft_signal, stft_template]

    def fft(self, x):
        x = x - np.median(x)
        x = np.pad(x, (0, 65536 - len(x)), mode='constant', constant_values=0)
        spectrums = sp_fft.rfft(x, axis=-1)
        # freqs = sp_fft.rfftfreq(spectrums.shape[-1], 1 / self.fs)
        return spectrums

#TODO нужно дополнительно откалибровать, разобраться с маштабом данных и zero padding
    def stft(self,x):
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
            f, t, Zxx = signal.stft(row, fs= self.fs, nperseg=wind, noverlap=noverlap)
            if freqs is None:
                freqs, times = f, t
            spectrograms.append(np.abs(Zxx))
        spectrograms = np.array(spectrograms)
        new_shape = original_shape[:-1] + (spectrograms.shape[1], spectrograms.shape[2])
        result = spectrograms.reshape(new_shape)

        return result

#блок эмбеддинга, отвечающий за соответсвие входящего сигнала и математической модели искажения сигнала
class MatchigFiltersEmbedding(nn.Module):

    def __init__(self):
        super(MatchigFiltersEmbedding, self).__init__()

    def

    def get_filters(self,template):
        filters = []


        return filters

    def forward(self,template, signal):
        filters = self.get_filters(template)

        pass



#Блок, очищающий входной снимок от лишних помех и позволяющий использовать исходный сигнал
#будет хорошо, если переедет в самое начала тракта очистки. Нужно тестить отдельно.

class DiffusionEmbedding(nn.Module):

    def __init__(self):
        super(DiffusionEmbedding, self).__init__()
        pass

class Embedding(nn.Module):

    def __init__(self, template, signal):
        super(Embedding, self).__init__()
        self.template = template
        self.signal = signal

        self.SpectralEmbeddingBlock = SpectralEmbeddingBlock(self.template, self.signal)
        self.MatchigFiltersEmbedding = MatchigFiltersEmbedding(self.template, self.signal)
        self.DiffusionEmbedding = DiffusionEmbedding(self.signal)

    def forward(self, template, signal):
        #TODO нужно поженить мнимые и действительные части тем или иным способом для разных fft и выводы фильтров
        pass
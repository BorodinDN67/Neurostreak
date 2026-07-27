import torch.nn as nn
import torch
import numpy as np


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


class SpectralMergeBlock(nn.Module):
    def __init__(self):
        super(SpectralMergeBlock, self).__init__()
    def forward(self, real, image):
        return torch.cat([real, image], 1)


#блок эмбеддинга, отвечающий за соответсвие входящего сигнала и математической модели искажения сигнала
# class MatchigFiltersEmbedding(nn.Module):
#
#     def __init__(self, config):
#         super(MatchigFiltersEmbedding, self).__init__()
#
#         self.config = config
#
#     def get_filters(self,template, config):
#         filters = []
#
#         water = Water(
#         absorption = config.absorption,
#         scattering = config.scattering,
#         indicatrix = config.indicatrix
#         )
#
#
#         for distance in range(1,40):
#             model = MTF_Dolin(
#                 water = water,
#                 k_theta = config.k_theta,
#                 k = config.k,
#                 L = config.L,
#                 mu = config.mu,
#                 theta0 = config.theta0,
#                 gamma = config.gamma,
#             )
#             filters.append(model.simulate(template, distance))
#
#         return filters
#
#     def forward(self, template, signal):
#         batch_size, lenght = signal.shape
#         template_norm = (template - np.median(template)) / np.std(template)
#         all_filters = []
#         res = [[] for _ in range(batch_size)]
#
#         for b in range(batch_size):
#             filters = self.get_filters(template_norm[b], self.config)
#             all_filters.append(filters)
#
#         num_filters = len(all_filters[0])
#
#         for b in range(batch_size):
#             signal_norm = (signal[b] - np.median(signal[b])) / np.std(signal[b])
#             for f_idx, filter in enumerate(all_filters[b]):
#                 corr = signal_.correlate(signal_norm, filter, mode='valid')
#                 corr /= len(filter)
#
#                 res[b].append(np.max(corr))
#         return torch.Tensor(scipy.special.softmax(res) )


# class STTFEmbeddingBlock(nn.Module):
#     def __init__(self):
#         super(STTFEmbeddingBlock, self).__init__()
#
#     def forward(self,x):
#         wind = 256
#         noverlap = wind // 2
#
#         spectrograms = []
#         freqs = None
#         times = None
#
#         if x.ndim == 1:
#             x = x[np.newaxis, :]
#         original_shape = x.shape
#         flattened_x = x.reshape(-1, x.shape[-1])
#
#         for row in flattened_x:
#             row = row - np.median(row)
#             row = np.pad(row, (0, 65535 - len(row)), mode='constant', constant_values=0)
#             f, t, Zxx = signal_.stft(row, fs= self.fs, nperseg=wind, noverlap=noverlap)
#             if freqs is None:
#                 freqs, times = f, t
#             spectrograms.append(np.abs(Zxx))
#         spectrograms = np.array(spectrograms)
#         new_shape = original_shape[:-1] + (spectrograms.shape[1], spectrograms.shape[2])
#         result = spectrograms.reshape(new_shape)
#
#         return result
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



#StreakNet //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class FDEmbeddingBlock(nn.Module):
    def __init__(self, width=1.0, act='silu', export=False):
        super(FDEmbeddingBlock, self).__init__()
        self.export = export
        self.embedding_size = round(512 * width)
        self.flatten = nn.Flatten(1)
        self.norm = nn.LayerNorm((4000 * 2,))
        self.dense = nn.Linear(4000 * 2, self.embedding_size)
        self.act = self.get_activation(act, inplace=False)

    def forward(self, x):
        """ export mode:
                x.shape:[batch, 2, 4000]
            else:
                x.shape:[batch, len]
        """
        if not self.export:
            # 满屏扫描时间30ns，CCD分辨率2048，采样频率68.27GHz
            # 使用长度65536计算FFT，频率分辨率68.27GHz/65536=1.04MHz
            signal_freq = torch.fft.rfft(x, 65536, dim=1)
            signal_freq = signal_freq[:, :4000]  # 只要0~4GHz
            signal_real = torch.real(signal_freq)
            signal_imag = torch.imag(signal_freq)
            concat_signal = torch.concat([signal_real, signal_imag], dim=1)
        else:
            concat_signal = x
        concat_signal = self.flatten(concat_signal).unsqueeze(1)
        pred = self.dense(self.norm(concat_signal))
        pred = self.act(pred)
        return pred

    def get_activation(self, name="silu", inplace=False):
        if name == "silu":
            module = nn.SiLU(inplace=inplace)
        elif name == "relu":
            module = nn.ReLU(inplace=inplace)
        elif name == "lrelu":
            module = nn.LeakyReLU(0.1, inplace=inplace)
        else:
            raise AttributeError("Unsupported act type: {}".format(name))
        return module


class FDEmbedding(nn.Module):
    def __init__(self, width=1.0, act='silu', export=False):
        super(FDEmbedding, self).__init__()
        self.signal_embedding_block = FDEmbeddingBlock(width, act, export)
        self.template_embedding_block = FDEmbeddingBlock(width, act, export)

    def forward(self, signal, template):
        signal_embedding = self.signal_embedding_block(signal)
        template_embedding = self.template_embedding_block(template)
        ret = torch.concat([signal_embedding, template_embedding], dim=1)
        return ret
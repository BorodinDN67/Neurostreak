import torch.nn as nn
from .head import StreakNetImagingHead
from .embedding import StreakNetEmbedding
from .backbone import StreakNetDBCAttention

from .head import NeuroStreakHead
from .backbone import NeuroStreakBackbone
from .embedding import NeuroStreakEmbedding


class NeurostreakArch(nn.Module):
    def __init__(self, embedding, backbone, head, config):
        super(NeurostreakArch, self).__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.head = head

        self.config = config

    def forward(self, signal, template):
        spectral_signal, spectral_template, wavelet_signal, wavelet_template = self.embedding(signal = signal, template = template)
        _, signal_wvlt_bkbn, _, signal_spctrl_bkbn = self.backbone(spectral_signal, spectral_template, wavelet_signal, wavelet_template)
        head = self.head(signal_spctrl_bkbn, signal_wvlt_bkbn)
        return head


class StreakNetArch(nn.Module):
    def __init__(self, embedding=None, backbone=None, head=None):
        super(StreakNetArch, self).__init__()
        if embedding is None:
            embedding = StreakNetEmbedding()
        if backbone is None:
            backbone = StreakNetDBCAttention()
        if head is None:
            backbone = StreakNetImagingHead()

        self.embedding = embedding
        self.backbone = backbone
        self.head = head

    def forward(self, signal, template, targets=None):
        embedding = self.embedding(signal = signal,template =  template)
        outs = self.backbone(embedding)

        if self.training:
            assert targets is not None
            outputs = self.head(outs, labels=targets)
        else:
            outputs = self.head(outs)

        return outputs

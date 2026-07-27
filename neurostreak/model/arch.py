import torch.nn as nn
from .head import StreakNetImagingHead
from .embedding import StreakNetEmbedding
from .backbone import StreakNetDBCAttention

class NeurostreakArch(nn.Module):
    def __init__(self, embedding, backbone, head, config):
        super(NeurostreakArch, self).__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.head = head

        self.config = config

    def forward(self, signal):
        embedding = self.embedding(signal)
        backbone = self.backbone(embedding)
        head = self.head(backbone)
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
        embedding = self.embedding(signal, template)
        outs = self.backbone(embedding)

        if self.training:
            assert targets is not None
            outputs = self.head(outs, labels=targets)
        else:
            outputs = self.head(outs)

        return outputs

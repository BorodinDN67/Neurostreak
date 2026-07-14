import torch.nn as nn


class NeurostreakArch(nn.Module):
    def __init__(self, embedding, backbone, head, config):
        super(NeurostreakArch, self).__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.head = head

        self.config = config

    def forward(self, signal):
        embedding = self.embedding(signal, config)
        backbone = self.backbone(embedding)
        head = self.head(backbone)
        return head
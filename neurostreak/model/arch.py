import torch.nn as nn


class NeurostreakArch(nn.Module):
    def __init__(self, embedding, backbone, head):
        super(NeurostreakArch, self).__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.head = head

    def forward(self, signal):
        embedding = self.embedding(signal)
        backbone = self.backbone(embedding)
        head = self.head(backbone)
        return head
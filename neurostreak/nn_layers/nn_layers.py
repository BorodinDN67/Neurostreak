import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(CrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cross_attention = nn.MultiheadAttention(embed_dim = self.embed_dim, num_heads=self.num_heads, batch_first=True)

    def forward(self, signal, template):
        query = signal
        key = template
        value = template

        return self.cross_attention(query, key, value)


class Convolution(nn.Module):
    def __init__(self):
        super(Convolution, self).__init__()
        pass
    def forward(self, query, key, value, mask):
        pass
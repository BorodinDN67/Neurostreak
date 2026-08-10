import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        wavelet_dim: int,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(wavelet_dim),
            nn.Linear(wavelet_dim, embed_dim),
        )

        self.adaptive_pool = AdaptivePooling(embed_dim, hidden_dim=128)

    def forward(self, x):
        projection = self.projector(x)
        # print('projection WaveletHead')
        # print(projection.shape)
        weights, scores = self.adaptive_pool(projection)
        # print('weights WaveletHead')
        # print(weights.shape)
        # print('scores WaveletHead')
        # print(scores.shape)
        return (weights.unsqueeze(-1)* projection).sum(dim=-2)


class SpectralHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        spectral_dim: int,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(spectral_dim),
            nn.Linear(spectral_dim, embed_dim),
        )

        self.adaptive_pool = AdaptivePooling(embed_dim, hidden_dim=128)

    def forward(self, x):
        projection = self.projector(x)
        # print('projection SpectralHead')
        # print(projection.shape)
        weights, scores = self.adaptive_pool(projection)
        # print('weights SpectralHead')
        # print(weights.shape)
        # print('scores SpectralHead')
        # print(scores.shape)
        return (weights.unsqueeze(-1)* projection).sum(dim=-2)



class AdaptivePooling(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()

        self.mask = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
        )

    def forward(self, x):
        scores = self.mask(x).squeeze(-1)
        weights = F.softmax(scores, dim=-1)

        return weights, scores
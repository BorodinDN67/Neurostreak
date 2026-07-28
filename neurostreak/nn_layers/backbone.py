from torch import nn
import torch

from .nn_layers import CrossAttention


class SpectralMergeBlock(nn.Module):
    def __init__(self):
        super(SpectralMergeBlock, self).__init__()
    def forward(self, real, image):
        return torch.cat([real, image], 1)


class WaveletAttentionBlock(nn.Module):
    def __init__(self,
            hidden_2d_dim_1: int,
            hidden_2d_dim_2: int,
            kernel_size: tuple[int, int],
            padding:  tuple[int, int],
            num_heads: int,
            embed_dim: int,
            depth: int,
            max_pool_kernel_size_1: tuple[int, int],
            max_pool_stride_1: tuple[int, int],
            max_pool_kernel_size_2: tuple[int, int],
            max_pool_stride_2: tuple[int, int],
            max_pool_kernel_size_3: tuple[int, int],
            max_pool_stride_3: tuple[int, int],
        ):
        super(WaveletAttentionBlock, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = hidden_2d_dim_1, kernel_size = kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = max_pool_kernel_size_1, stride = max_pool_stride_1),

            nn.Conv2d(in_channels = hidden_2d_dim_1, out_channels = hidden_2d_dim_2, kernel_size = kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = max_pool_kernel_size_2, stride = max_pool_stride_2),

            nn.Conv2d(in_channels=hidden_2d_dim_2, out_channels=embed_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pool_kernel_size_3, stride=max_pool_stride_3),
        )
        self.cross_attention = nn.ModuleList([
            CrossAttention(num_heads=num_heads, embed_dim = embed_dim)
        for _ in range(depth)
        ])

    def forward(self, embedding_template, embedding_signal):

        template_conv = self.convolution(embedding_template)
        signal_conv = self.convolution(embedding_signal)

        cross_attention = signal_conv
        for layer in self.cross_attention:
            cross_attention = layer(template = template_conv, signal =  cross_attention)

        return template_conv, cross_attention


#StreakNet backbone layer //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DBCAttentionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: nn.Module):
        super(DBCAttentionLayer, self).__init__()

        self.signal_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.template_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.signal_addnorm1 = AddNorm(normalized_shape=(d_model,), dropout=dropout)
        self.signal_feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.signal_addnorm2 = AddNorm(normalized_shape=(d_model,), dropout=dropout)
        self.signal_act = activation

        self.template_addnorm1 = AddNorm(normalized_shape=(d_model,), dropout=dropout)
        self.template_feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.template_addnorm2 = AddNorm(normalized_shape=(d_model,), dropout=dropout)
        self.template_act = activation

    def forward(self, signal, template):
        signal_aten, _ = self.signal_attention(signal, template, template, need_weights=False)
        template_aten, _ = self.template_attention(template, signal, signal, need_weights=False)

        signal_aten_norm = self.signal_addnorm1(signal, signal_aten)
        signal_feedforward = self.signal_feedforward(signal_aten_norm)
        signal_feed_norm = self.signal_addnorm2(signal_feedforward, signal_aten_norm)
        signal_output = self.signal_act(signal_feed_norm)

        template_aten_norm = self.template_addnorm1(template, template_aten)
        template_feedforward = self.template_feedforward(template_aten_norm)
        template_feed_norm = self.template_addnorm2(template_feedforward, template_aten_norm)
        template_output = self.template_act(template_feed_norm)

        return signal_output, template_output
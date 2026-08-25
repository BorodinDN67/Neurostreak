from torch import nn
import torch
from torch.onnx.ops import attention

from .nn_layers import CrossAttention


class SpectralMergeBlock(nn.Module):
    def __init__(self):
        super(SpectralMergeBlock, self).__init__()
    def forward(self, real, image):
        return torch.cat([real, image], 1)


class WaveletAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, depth ):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = hidden_dim // 4, kernel_size = (1, 7), padding=(0,3), stride = (1,3)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = hidden_dim// 4, out_channels = hidden_dim // 2, kernel_size = (1, 3), padding=(0,1), stride = (1,2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = (2,1), stride = 1),
        )
        self.embedding_projection = nn.Linear(in_features = 1, out_features = 1)

        self.cross_attention = nn.ModuleList([
            CrossAttention(num_heads=num_heads, embed_dim = 784)
        for _ in range(depth)
        ])

    def forward(self, embedding_template, embedding_signal):

        template_conv = self.convolution(embedding_template)
        signal_conv = self.convolution(embedding_signal)
        tmp = torch.flatten(template_conv, start_dim = -3, end_dim=-2).transpose(1, 2)
        signal = torch.flatten(signal_conv, start_dim = -3, end_dim=-2).transpose(1, 2)
        # print(signal_conv.shape)
        # print(signal.shape)


        for layer in self.cross_attention:
            attention_out = layer(template=tmp, signal=signal)
            signal = signal + attention_out

        return tmp, signal

class SpectralAttentionBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        depth: int
    ):
        super().__init__()
        self.cross_attention = nn.ModuleList([
            CrossAttention(num_heads=num_heads, embed_dim = embed_dim)
        for _ in range(depth)
        ])

        self.features = nn.Conv1d(in_channels = 1, out_channels = embed_dim, kernel_size = 11, padding = 5)

    def forward(self, embedding_template, embedding_signal):

        template_features = self.features(embedding_template)
        signal_features = self.features(embedding_signal)
        template_view = template_features.transpose(1, 2)
        signal_view = signal_features.transpose(1, 2)


        for layer in self.cross_attention:
            attention_out = layer(template = template_view, signal = signal_view)
            signal_view = signal_view + attention_out
        return template_view, signal_view
































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
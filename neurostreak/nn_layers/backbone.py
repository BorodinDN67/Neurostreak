from torch import nn
import torch

class SpectralMergeBlock(nn.Module):
    def __init__(self):
        super(SpectralMergeBlock, self).__init__()
    def forward(self, real, image):
        return torch.cat([real, image], 1)


class WaveletAttentionBlock(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding:int,
        ):
        super(WaveletAttentionBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.CrossAttention = nn.MultiheadAttention(embed_dim=512, num_heads=2)

    def forward(self, embedding_image, embedding_signal):
        pass


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
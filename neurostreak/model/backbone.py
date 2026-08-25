import torch.nn as nn
import torch
import copy



from ..nn_layers import WaveletAttentionBlock, DBCAttentionLayer, SpectralAttentionBlock
# Данные из фурье - > кросс атеншн между собой
# Данные из вейвлет -> кросс атенншн между собой
# осмысление и принятие решения происходит в голове

class NeuroStreakBackbone(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            depth: int,

    ):
        super().__init__()
        self.wavelet_attention_block = WaveletAttentionBlock(
            hidden_dim,
            num_heads,
            depth,
        )

        self.signal_attention_block = SpectralAttentionBlock(
            num_heads=num_heads,
            embed_dim=64,
            depth=depth,
        )

    def forward(self, spectral_signal, spectral_template, wavelet_signal, wavelet_template):


        template_wvlt_bkbn, signal_wvlt_bkbn = self.wavelet_attention_block(embedding_signal = wavelet_signal.unsqueeze(1),embedding_template = wavelet_template.unsqueeze(1))
        template_spctrl_bkbn, signal_spctrl_bkbn = self.signal_attention_block(embedding_signal = spectral_signal,embedding_template = spectral_template)

        return template_wvlt_bkbn, signal_wvlt_bkbn, template_spctrl_bkbn, signal_spctrl_bkbn

class StreakNetDBCAttention(nn.Module):
    def __init__(self, width: float = 1.00, depth: float = 1.00, dropout: float = 0.4, act: str = 'silu'):
        super(StreakNetDBCAttention, self).__init__()

        attention_layer = DBCAttentionLayer(
            d_model=round(512 * width),
            nhead=round(16 * width),
            dim_feedforward=round(512 * 2 * width),
            dropout=dropout,
            activation=self.get_activation(act, False)
        )
        self.layers = self._get_clones(attention_layer, N=round(8 * depth))

    def forward(self, x):
        out_sig = x[:, :1, :]
        out_tem = x[:, 1:, :]
        for layer in self.layers:
            out_sig, out_tem = layer(out_sig, out_tem)
        pred = torch.concat([out_sig, out_tem], dim=1)
        return pred


    def _get_clones(self,module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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
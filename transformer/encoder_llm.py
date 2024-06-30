from .encoder import Encoder
from torch import nn, Tensor


class EncoderLLM(nn.Module):
    def __init__(self, corpus_size: int, d_model: int, d_attn: int, d_ff: int, n_heads: int, n_layers: int):
        super(EncoderLLM, self).__init__()
        self.encoder = Encoder(corpus_size, d_model, d_attn, d_ff, n_heads, n_layers)
        self.o = nn.Linear(d_model, corpus_size)

    def forward(self, x: Tensor, mask: Tensor = None):
        # mean = .mean(dim=1)
        return self.o(self.encoder(x, mask))


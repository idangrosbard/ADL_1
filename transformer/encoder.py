import torch
from torch import nn
from torch import functional as F
from .attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_attn: int, n_heads: int, d_ff: int):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model

        self.attention = MultiHeadAttention(d_model, d_attn, n_heads, cross_attention=False)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None):
        attention = self.attention(x, cross=None, mask=mask)
        x = x + attention
        x = F.layer_norm(x, x.size()[1:])
        x = x + self.feedforward(attention)
        x = F.layer_norm(x, x.size()[1:])
        return x


class Encoder(nn.Module):
    def __init__(self, corpus_size: int, d_model: int, d_attn: int, d_ff: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(corpus_size, d_model)
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_attn, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

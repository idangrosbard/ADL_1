import torch
from torch import nn
from torch import functional as F
from .attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_attn: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(d_model, d_attn, n_heads, cross_attention=False)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attention = self.attention(x, cross=None, mask=mask)
        x = x + attention
        x = self.layer_norm(x)
        x = x + self.feedforward(attention)
        x = self.layer_norm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, corpus_size: int, d_model: int, d_attn: int, d_ff: int, n_heads: int, n_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed = nn.Embedding(corpus_size, d_model, corpus_size - 1)
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_attn, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x, mask)
        return x

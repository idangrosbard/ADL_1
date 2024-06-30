import torch
from torch import nn, Tensor
from torch import functional as F
from .attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_attn: int, n_heads: int, d_ff: int):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model

        self.self_attention = MultiHeadAttention(d_attn, n_heads, cross_attention=False)
        self.cross_attention = MultiHeadAttention(d_attn, n_heads, cross_attention=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: Tensor, cross: Tensor):
        self_attention = self.self_attention(x)
        x = x + self_attention
        x = F.layer_norm(x, x.size()[1:])
        
        cross_attention = self.cross_attention(x, cross)
        x = x + cross_attention
        x = F.layer_norm(x, x.size()[1:])
        
        x = x + self.feedforward(x)
        x = F.layer_norm(x, x.size()[1:])
        return x


class Encoder(nn.Module):
    def __init__(self, corpus_size: int, d_model: int, d_attn: int, d_ff: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(corpus_size, d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_attn, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, x: Tensor, cross: Tensor):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, cross)
        return x

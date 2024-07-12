import torch
from torch import nn, Tensor
from torch import functional as F
from .attention import MultiHeadAttention
from typing import Optional


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_attn: int, n_heads: int, d_ff: int, dropout: float = 0.5, decoder_only: bool = True):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.decoder_only = decoder_only

        if not decoder_only:
            self.self_attention = MultiHeadAttention(d_model, d_attn, n_heads, cross_attention=False)
        self.cross_attention = MultiHeadAttention(d_model, d_attn, n_heads, cross_attention=True)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: Tensor, cross: Optional[Tensor], mask: Tensor):
        if not self.decoder_only:
            self_attention = self.self_attention(x, mask=mask)
            x = x + self_attention
        
        cross_attention = self.cross_attention(x, cross=cross, mask=mask)
        x = x + cross_attention
        
        x = x + self.feedforward(x)
        return x


class Decoder(nn.Module):
    def __init__(self, corpus_size: int, d_model: int, d_attn: int, d_ff: int, n_heads: int, n_layers: int, output_dim: int, decoder_only: bool = True, return_logits: bool = True) -> None:
        super().__init__()
        self.return_logits = return_logits
        self.embed = nn.Embedding(corpus_size, d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_attn, n_heads, d_ff, decoder_only=decoder_only) for _ in range(n_layers)])
        self.o = nn.Linear(d_model, output_dim)

    def forward(self, x: Tensor, cross: Optional[Tensor], mask: Tensor):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, cross, mask)

        if self.return_logits:
            return self.o(x)
        return x

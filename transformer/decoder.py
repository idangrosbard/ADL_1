import torch
from torch import nn, Tensor
from torch import functional as F
from .attention import MultiHeadAttention
from typing import Optional
from .pe import PositionalEncoding


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_attn: int, n_heads: int, d_ff: int, dropout: float = 0.1, decoder_only: bool = True):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.decoder_only = decoder_only

        self.causal_self_attention = MultiHeadAttention(d_model, d_attn, n_heads, cross_attention=True)
        self.cross_attention = MultiHeadAttention(d_model, d_attn, n_heads, cross_attention=True)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.causal_dropout = nn.Dropout(dropout)
        self.cross_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

        self.causal_layer_norm = nn.LayerNorm(d_model)
        self.cross_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)


    def forward(self, x: Tensor, cross: Optional[Tensor], mask: Tensor):
        self_attention = self.causal_self_attention(x, mask=mask)
        self_attention = self.causal_dropout(self_attention)
        x = x + self_attention
        x = self.causal_layer_norm(x)
        
        cross_attention = self.cross_attention(x, cross=cross, mask=mask)
        cross_attention = self.cross_dropout(cross_attention)
        x = x + cross_attention
        x = self.cross_layer_norm(x)
        
        ff = self.feedforward(x)
        ff = self.ff_dropout(ff)
        x = x + ff
        x = self.ff_layer_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, corpus_size: int, d_model: int, d_attn: int, d_ff: int, n_heads: int, n_layers: int, output_dim: int, decoder_only: bool = True, return_logits: bool = True, max_l: int = 10000) -> None:
        super().__init__()
        print(corpus_size)
        self.return_logits = return_logits
        self.embed = nn.Embedding(corpus_size, d_model, padding_idx=28439)
        self.pe = PositionalEncoding(d_model, max_l)
        self.d_model = d_model

        self.layers = nn.ModuleList([DecoderBlock(d_model, d_attn, n_heads, d_ff, decoder_only=decoder_only) for _ in range(n_layers)])
        # self.normalizers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.o = nn.Linear(d_model, output_dim)

    def forward(self, x: Tensor, cross: Optional[Tensor], mask: Tensor):
        x = self.embed(x)
        
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, cross, mask)
        
        if self.return_logits:
            return self.o(x)
        return x

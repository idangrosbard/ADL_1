import torch
from torch import nn, Tensor
from torch import functional as F
from typing import Optional


class Attention(nn.Module):
    def __init__(self, d_model: int, d_attn: int, cross_attention: bool = False):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.d_attn = d_attn
        self.cross_attention = cross_attention

        self.q = nn.Linear(d_model, d_attn, bias=False)
        self.k = nn.Linear(d_model, d_attn, bias=False)
        self.v = nn.Linear(d_model, d_attn, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.o = nn.Linear(d_attn, d_model)

    
    def forward(self, x: Tensor, cross: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        if cross is not None:
            q = self.q(x)
            k = self.k(cross)
        else:
            q = self.q(x)
            k = self.k(x)
        
        v = self.v(x)

        logits = q @ k.transpose(-2, -1)
        
        if self.cross_attention:
            mask = torch.triu(torch.ones_like(logits), diagonal=1)
            logits[mask] = float('-inf')

        if mask is not None:
            logits[mask == 0] = float('-inf')

        attn = self.softmax(logits / (self.d_model ** 0.5))

        attn_out = attn @ v
        # print(o.shape)
        return self.o(attn_out)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, attn_dim: int, n_heads: int, cross_attention: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.attentions = nn.ModuleList([Attention(d_model, attn_dim, cross_attention) for _ in range(n_heads)])
        self.o = nn.Linear(d_model * n_heads, d_model)
    
    def forward(self, x: Tensor, cross: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        multi_attention = torch.cat([attention(x, cross=cross, mask=mask) for attention in self.attentions], dim=-1)
        return self.o(multi_attention)

        

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
        # shape of X [b, l_x, d]
        # shape of cross [b, l_cross, d]
        # shape of mask [b, l_x]
        if cross is None:
            cross = x
        
        q = self.q(x) # [b, l_x, d]
        k = self.k(cross) # [b, l_cross, d]
        
        v = self.v(x) # [b, l_x, d]

        logits = q @ k.transpose(-2, -1) # [b, l_x, l_cross]
        
        if self.cross_attention:
            triu_mask = torch.triu(torch.ones_like(logits), diagonal=1) # [b, l_x, l_cross]
            logits = logits.where(triu_mask == 0, float('-inf')) # set -inf for the upper triangle

        if mask is not None:
            square_mask = mask.unsqueeze(-1)
            square_mask = square_mask @ square_mask.transpose(-1, -2) # [b, l_x, l_x]
            logits = logits.where(square_mask == 1, float('-inf')) # set -inf to the masked positions

        attn = self.softmax(logits / (self.d_model ** 0.5)) # get distribution

        attn_out = attn @ v # get new reps [b, l_x, d]
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

        

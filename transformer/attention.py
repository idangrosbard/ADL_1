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
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    
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
            del triu_mask

        if mask is not None:
            square_mask = mask.unsqueeze(-1).float()
            square_mask = square_mask @ square_mask.transpose(-1, -2) # [b, l_x, l_x]
            # Add main diagonal to the mask (to support padding tokens)
            square_mask = square_mask + torch.eye(mask.shape[1], device=mask.device).unsqueeze(0)
            logits = logits.where(square_mask > 0, float('-inf')) # set -inf to the masked positions
            del square_mask

        attn = self.softmax(logits / (self.d_attn ** 0.5)) # get distribution
        del logits

        attn_out = attn @ v # get new reps [b, l_x, d]
        del attn
        assert attn_out.shape == x.shape
        return attn_out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, attn_dim: int, n_heads: int, cross_attention: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.attentions = nn.ModuleList([Attention(d_model, attn_dim, cross_attention) for _ in range(n_heads)])
        self.o = nn.Linear(d_model * n_heads, d_model)
    
    def forward(self, x: Tensor, cross: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        multi_attention = torch.cat([attention(x, cross=cross, mask=mask) for attention in self.attentions], dim=-1)
        assert multi_attention.shape == (x.shape[0], x.shape[1], self.d_model * self.n_heads)
        out = self.o(multi_attention)
        assert out.shape == x.shape
        return out

        

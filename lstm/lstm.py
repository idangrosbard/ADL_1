import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple, List


class LSTM(nn.Module):
    def __init__(self, d_input: int, d_hidden: int):
        super(LSTM, self).__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden

        self.forget = nn.Linear(d_input + d_hidden, d_hidden)
        self.cell = nn.Linear(d_input + d_hidden, d_hidden)
        self.magnitude = nn.Linear(d_input + d_hidden, d_hidden)
        self.output = nn.Linear(d_input + d_hidden, d_hidden)


    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        f = F.sigmoid(self.forget(torch.concat([x, h], dim=-1)))
        c_f = c * f

        c_t = F.tanh(self.cell(torch.concat([x, h], dim=-1)))
        m_t = F.sigmoid(self.magnitude(torch.concat([x, h], dim=-1)))  
        c = c_f + c_t * m_t

        o = F.sigmoid(self.output(torch.concat([x, h], dim=-1)))
        h = o * F.tanh(c)
        return h, c
    
    
class LSTMBlock(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, n_layers: int, dropout: float = 0.5):
        super(LSTMBlock, self).__init__()
        self.d_input = d_input
        
        self.block = nn.ModuleList([LSTM(d_input if i == 0 else d_hidden, d_hidden) for i in range(n_layers)])
        
        self.mlps = nn.ModuleList([
            nn.Sequential(*[nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(d_hidden)]) 
            for i in range(n_layers)])

    def forward(self, x: Tensor, hs: List[Tensor], cs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        new_hs, new_cs = [], []
        h = x
        for i, lstm in enumerate(self.block):
            h, c = lstm(h, hs[i], cs[i])
            new_hs.append(h)
            h = self.mlps[i](h)
            new_cs.append(c)
        return new_hs, new_cs

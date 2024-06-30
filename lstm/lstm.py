import torch
from torch import nn, Tensor
from torch import functional as F
from typing import Tuple


class LSTM(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_layers: int):
        super(LSTM, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        self.forget = nn.Linear(d_model + d_hidden, d_hidden)
        self.cell = nn.Linear(d_model + d_hidden, d_hidden)
        self.magnitude = nn.Linear(d_model + d_hidden, d_hidden)
        self.output = nn.Linear(d_model + d_hidden, d_model)


    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        f = F.sigmoid(self.forget(torch.concat([x, h], dim=-1)))
        c_f = c * f

        c_t = F.tanh(self.cell(torch.concat([x, h], dim=-1)))
        m_t = F.sigmoid(self.magnitude(torch.concat([x, h], dim=-1)))  
        c = c_f + c_t * m_t

        o = F.sigmoid(self.output(torch.concat([x, h], dim=-1)))
        h = o * F.tanh(c)
        return h, c
    
    
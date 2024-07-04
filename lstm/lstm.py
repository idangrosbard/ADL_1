import torch
from torch import nn, Tensor
from torch import functional as F
from typing import Tuple, List


class LSTM(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super(LSTM, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

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
    
    
class LSTMBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_layers: int):
        super(LSTMBlock, self).__init__()
        self.d_model = d_model
        self.block = nn.ModuleList([LSTM(d_model, d_hidden) for _ in range(n_layers)])

    def forward(self, x: Tensor, hs: List[Tensor], cs: List[Tensor]) -> Tuple[Tensor, Tensor]:
        new_hs, new_cs = [], []
        for i, lstm in enumerate(self.block):
            h, c = lstm(x, self.hiddens[i], self.cells[i])
            new_hs.append(h)
            new_cs.append(c)
        return x, new_hs, new_cs

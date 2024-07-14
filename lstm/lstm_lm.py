from.lstm import LSTMBlock
from torch import nn, Tensor, tensor
from typing import Optional, Tuple, List


class LSTM_LM(nn.Module):
    def __init__(self, corpus_size: int, d_hidden: int, n_layers: int):
        super(LSTM_LM, self).__init__()
        self.d_hidden = d_hidden
        self.embed = nn.Embedding(corpus_size, d_hidden, padding_idx=28439)
        self.lstm_blocks = nn.ModuleList([LSTMBlock(d_hidden, d_hidden) for i in range(n_layers)])
        self.o = nn.Linear(d_hidden, corpus_size)
        self.return_logits = True
    
    def forward(self, x: Tensor, hs: Optional[List[Tensor]], cs: Optional[List[Tensor]]) -> Tuple[Tensor, List[Tensor], List[Tensor]]:   
        x = self.embed(x)
        new_hs = []
        new_cs = []
        if hs is None:
          for i in range(len(self.lstm_blocks)):
              new_h, new_c = self.lstm_blocks[i](x, None, None)
              new_hs.append(new_h)
              new_cs.append(new_c)
        else:
          for block, h, c in zip(self.lstm_blocks, hs, cs):
              new_h = x
              new_h, new_c = block(new_h, h, c)
              
              new_hs.append(new_h)
              new_cs.append(new_c)
        
        if self.return_logits:
            return self.o(new_h), new_hs, new_cs
        else:
            return new_h, new_hs, new_cs
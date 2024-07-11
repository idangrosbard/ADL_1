from.lstm import LSTMBlock
from torch import nn, Tensor
from typing import Optional, Tuple, List


class LSTM_LM(nn.Module):
    def __init__(self, corpus_size: int, d_hidden: int, n_layers: int):
        super(LSTM_LM, self).__init__()

        self.embed = nn.Embedding(corpus_size, d_hidden, corpus_size - 1)
        self.lstm_blocks = nn.ModuleList([LSTMBlock(d_hidden, d_hidden) for i in range(n_layers)])
        self.o = nn.Linear(d_hidden, corpus_size)
    
    def forward(self, x: Tensor, hs: Optional[Tensor], cs: Optional[Tensor]) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        x = self.embed(x)
        new_hs = []
        new_cs = []
        if hs is None:
          for i in range(len(self.lstm_blocks)):
              h, c = self.lstm_blocks[i](x, None, None)
              new_hs.append(h)
              new_cs.append(c)
        else:
          for block, h, c in range(zip(self.lstm_blocks, hs, cs)):
              new_h = x
              new_h, new_c = block(new_h, h, c)
              
              new_hs.append(h)
              new_cs.append(c)
        
        return self.o(hs[-1]), new_hs, new_cs
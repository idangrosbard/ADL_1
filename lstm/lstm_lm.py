from.lstm import LSTMBlock
from torch import nn, Tensor
from typing import Optional, Tuple, List


class LSTM_LM(nn.Module):
    def __init__(self, corpus_size: int, d_input: int, d_hidden: int, n_layers: int):
        super(LSTM_LM, self).__init__()

        self.embed = nn.Embedding(corpus_size, d_input, corpus_size - 1)
        self.lstm_blocks = nn.ModuleList([LSTMBlock(d_input if i == 0 else d_hidden, d_hidden) for i in range(n_layers)])
        self.o = nn.Linear(d_hidden, corpus_size)
    
    def forward(self, x: Tensor, hs: Optional[Tensor], cs: Optional[Tensor]) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        x = self.embed(x)
        hs, cs = [], []
        for i in range(len(self.lstm_blocks)):
            if hs is not None:
                h, c = self.lstm_blocks[i](x, hs[i], cs[i])
            else:
                h, c = self.lstm_blocks[i](x, None, None)
            hs.append(h)
            cs.append(c)
        
        return self.o(hs[-1]), hs, cs
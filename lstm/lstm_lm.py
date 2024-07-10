from.lstm import LSTMBlock
from torch import nn, Tensor
from typing import Optional, Tuple, List


class LSTM_LM(nn.Module):
    def __init__(self, corpus_size: int, d_input: int, d_hidden: int, n_layers: int):
        super(LSTM_LM, self).__init__()

        self.embed = nn.Embedding(corpus_size, d_input, corpus_size - 1)
        self.lstm_blocks = LSTMBlock(d_input, d_hidden, n_layers)
        self.o = nn.Linear(d_hidden, corpus_size)
        
        self.h0s = [nn.Parameter(Tensor(d_hidden)) for _ in range(n_layers)]
        self.c0s = [nn.Parameter(Tensor(d_hidden)) for _ in range(n_layers)]
    
    def forward(self, x: Tensor, hs: Optional[Tensor], cs: Optional[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        x = self.embed(x)
        b_size = x.shape[0]
        if hs is None:
            hs = [h0.repeat((b_size,1)) for h0 in self.h0s]
        if cs is None:
            cs = [c0.repeat((b_size,1)) for c0 in self.c0s]
        
        hs, cs = self.lstm_blocks(x, hs, cs)
        
        return self.o(hs[-1]), hs, cs
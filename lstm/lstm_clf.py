from torch import nn, Tensor, tensor
from typing import Tuple, List
from .lstm_lm import LSTM_LM


class LSTMClassifier(nn.Module):
    def __init__(self, lstm_lm: LSTM_LM, d_hidden: int,  n_classes: int):
        super(LSTMClassifier, self).__init__()

        self.lstm = lstm_lm
        self.fc = nn.Linear(d_hidden, n_classes)
    
    def forward(self, x: Tensor, attn: Tensor) -> Tensor:
        hs = None
        cs = None
        hs_through_time = []
        # Iterate over the sequence, get the last hidden state of the last input token
        for t in range(x.shape[1]):
            _, hs, cs = self.lstm(x[:,t], hs, cs)
            # store h history per token, to get the last hidden state per each sample (supporting different lengths)
            hs_through_time.append(hs[-1])

        # Get the last hidden state of the last input token
        lengths = attn.sum(dim=1)
        x = self.lstm(tensor(hs_through_time)[lengths])
        
        logits = self.fc(x)
        return logits
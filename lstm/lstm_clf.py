
from torch import nn, Tensor, stack
from typing import Tuple, List
from .lstm_lm import LSTM_LM


class LSTMClassifier(nn.Module):
    def __init__(self, lstm_lm: LSTM_LM, d_hidden: int,  n_classes: int):
        super(LSTMClassifier, self).__init__()

        self.lstm = lstm_lm
        self.lstm.return_logits = False
        self.fc = nn.Linear(d_hidden, n_classes)
    
    def forward(self, x: Tensor, attn: Tensor) -> Tensor:
        hs = None
        cs = None
        hs_through_time = []
        # Iterate over the sequence, get the last hidden state of the last input token
        for t in range(x.shape[1]):
            h, hs, cs = self.lstm(x[:,t], hs, cs)
            # store h history per token, to get the last hidden state per each sample (supporting different lengths)
            hs_through_time.append(h)

        # Get the last hidden state of the last input token
        lengths = attn.sum(dim=1) - 1
        # print(attn.shape, l, l.shape)
        # print(len(hs_through_time))
        # print(len(hs_through_time[0].shape))
        last_hs = []
        for b_idx in range(lengths.shape[0]):
            l = lengths[b_idx].long()
            h_at_time_l = hs_through_time[l]
            h_of_sample_idx = h_at_time_l[b_idx]
            last_hs.append(h_of_sample_idx)
        
        last_hs = stack(last_hs).to(self.fc.weight.device)
        
        logits = self.fc(last_hs)
        return logits
from torch import nn, Tensor
from typing import Tuple, List
from .lstm_lm import LSTM_LM


class LSTMClassifier(nn.Module):
    def __init__(self, lstm_lm: LSTM_LM, d_hidden: int,  n_classes: int):
        super(LSTMClassifier, self).__init__()

        self.lstm = lstm_lm
        self.fc = nn.Linear(d_hidden, n_classes)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        return self.fc(self.lstm(x))
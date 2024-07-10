from torch import nn, Tensor
from .s4_model import S4Model



class S4Classifier(nn.Module):
    def __init__(self, s4_model: S4Model, d_hidden: int, n_classes: int):
        super(S4Classifier, self).__init__()

        self.s4 = s4_model
        self.fc = nn.Linear(d_hidden, n_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.s4(x[:, -1])) # Take the last token, assume x.shape = [batch, seq_len, d_hidden]
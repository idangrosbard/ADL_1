from torch import nn, Tensor
from .s4_model import S4Model



class S4Classifier(nn.Module):
    def __init__(self, s4_model: S4Model, d_hidden: int, n_classes: int):
        super(S4Classifier, self).__init__()
        self.s4 = s4_model
        self.s4.return_logits = False
        self.fc = nn.Linear(d_hidden, n_classes)
    
    def forward(self, x: Tensor, attn: Tensor) -> Tensor:
        x = self.s4(x)
        # Get the last hidden state of the last input token
        L = attn.sum(dim=1) - 1
        last_x = x.gather(1, L.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2))).squeeze(1)
        y = self.fc(last_x)
        return y.squeeze(1)
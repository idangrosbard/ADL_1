from torch import nn
from .encoder_llm import EncoderLLM


class EncoderClf(nn.Module):
    def __init__(self, lm: EncoderLLM, d_hidden: int, n_cls: int) -> None:
        super().__init__()
        self.lm = lm
        self.fc = nn.Linear(d_hidden, n_cls)

    def forward(self, x):
        x = self.lm(x)
        x = x.mean(dim=1) # assume x.shape = [batch, seq_len, d_hidden]
        return self.fc(x)
    


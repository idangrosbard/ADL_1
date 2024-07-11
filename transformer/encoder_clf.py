from torch import nn, Tensor
from .encoder_llm import EncoderLLM


class EncoderClf(nn.Module):
    def __init__(self, lm: EncoderLLM, d_hidden: int, n_cls: int) -> None:
        super().__init__()
        self.lm = lm
        self.fc = nn.Linear(d_hidden, n_cls)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        x = self.lm(x, attention_mask)
        x = x * attention_mask # Filter out padding tokens
        x = x.sum(dim=1) / attention_mask.sum(dim=1) # Average over non-padding tokens
        return self.fc(x)
    


from torch import nn, Tensor
from .decoder import Decoder


class DecoderClf(nn.Module):
    def __init__(self, lm: Decoder, d_hidden: int, n_cls: int) -> None:
        super().__init__()
        self.lm = lm
        self.lm.return_logits = False
        self.fc = nn.Linear(d_hidden, n_cls)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        x = self.lm(x, None, attention_mask)
        # get the last token per sample in batch
        last_token_idx = (attention_mask.sum(dim=1) - 1).long()
        x = x.gather(1, last_token_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, x.size(2))).squeeze(1)
        del last_token_idx
        return self.fc(x)
    


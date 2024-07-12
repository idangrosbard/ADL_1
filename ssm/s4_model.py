import torch
from torch import nn, Tensor
from .s4d_layer import S4DLayer
from .pe import PositionalEncoding

class S4Block(nn.Module):
    def __init__(self, H: int, N: int, dropout: float = 0.5):
        super().__init__()
        
        self.layers = nn.Sequential(
            S4DLayer(H, N),
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(dropout),)
        
    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)


class S4Model(nn.Module):
    def __init__(self, H: int, N: int, vocab_size: int, output_dim: int, N_layers: int, dropout: float = 0.5, max_l: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.N_layers = N_layers
        self.emb = nn.Embedding(vocab_size, H, padding_idx=28440)
        self.pe = PositionalEncoding(H, max_l)
        self.use_token_clf = True
        
        layers = [S4Block(H, N, dropout) for _ in range(N_layers)]
        self.layers = nn.Sequential(*layers)

        self.out = nn.Linear(H, output_dim)

    
    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        x = self.pe(x)
        x = self.layers(x)
        
        if self.use_token_clf:
            x = self.out(x)
        return x
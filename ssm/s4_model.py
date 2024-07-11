import torch
from torch import nn, Tensor
from .s4d_layer import S4DLayer



class S4Model(nn.Module):
    def __init__(self, H: int, N: int, vocab_size: int, output_dim: int, N_layers: int, dropout: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.N_layers = N_layers
        self.emb = nn.Embedding(vocab_size, H)
        self.use_token_clf = True
        
        layers = []
        for _ in range(N_layers):
            layers.append(nn.Linear(H, H))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(S4DLayer(H, N))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.layers = nn.ModuleList(layers)

        self.out = nn.Linear(H, output_dim)

    
    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x) + x
        
        if self.use_token_clf:
            x = self.out(x)
        return x
import torch
from torch import nn, fft, tensor, Tensor
import numpy as np


class S4DLayer(nn.Module):
    def __init__(self, H: int, N: int, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        self.H = H
        self.N = N
        self.log_delta = nn.Parameter(torch.rand((H, 1), dtype=torch.float32) * (torch.log(tensor(dt_max)) - torch.log(tensor(dt_min))) + torch.log(tensor(dt_min)))
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones((H, N // 2), dtype=torch.float32)))
        self.A_imag = nn.Parameter(torch.tensor(torch.pi * torch.arange(N // 2).repeat((H,1)), dtype=torch.float32))
        self.B = nn.Parameter(torch.ones((H, N // 2), dtype=torch.float32))
        self.C = nn.Parameter(torch.view_as_real(torch.randn((H, N // 2), dtype=torch.complex32)))
        self.D = nn.Parameter(torch.randn((H), dtype=torch.float32))
        
    
    def kernel(self, L: int):
        delta = torch.exp(self.log_delta)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dA = (1 + delta * A / 2) / (1 - delta * A / 2)
        K = 2 * torch.einsum('hn,hnl->hl', self.B * torch.view_as_complex(self.C), (dA.unsqueeze(-1) ** torch.arange(L, device=dA.device))).real

        del delta, A, dA

        return K
        
    def forward(self, u: Tensor): # Shape of u: (batch_size, L, H)
        L = u.shape[-2]
        # u = u.squeeze(0)
        u = u.transpose(-2,-1)
        K = self.kernel(L)

        K_f = fft.rfft(K, n=2*L)
        u_f = fft.rfft(u, n=2*L)

        y = fft.irfft(K_f * u_f, n=2*L)[..., :L]
        y = y + u * self.D.unsqueeze(-1)
        y = y.transpose(-2,-1)

        del K, K_f, u_f

        return y
    
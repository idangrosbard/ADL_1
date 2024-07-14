import torch
from torch import nn, fft, tensor, Tensor
import numpy as np


class S4DLayer(nn.Module):
    def __init__(self, H: int, N: int, dt_min=1e-3, dt_max=1e-1, init_method='zoh'):
        super().__init__()
        self.H = H
        self.N = N
        self.init_method = init_method
        log_delta = torch.rand((H, 1), dtype=torch.float32) * (torch.log(tensor(dt_max)) - torch.log(tensor(dt_min))) + torch.log(tensor(dt_min))
        self.register_buffer('log_delta', log_delta)

        log_A_real = torch.log(0.5 * torch.ones((H, N // 2), dtype=torch.float32))
        A_imag = torch.pi * torch.arange(N // 2, dtype=torch.float32).repeat((H,1))
        self.register_buffer('A_imag', A_imag)
        self.register_buffer('log_A_real', log_A_real)
        
        self.C = nn.Parameter(torch.view_as_real(torch.randn((H, N // 2), dtype=torch.cfloat)))
        if self.init_method == 'lin':
            self.B = nn.Parameter(torch.ones((H, N // 2), dtype=torch.float32))
        self.D = nn.Parameter(torch.randn((H), dtype=torch.float32))
        
    
    def kernel_zoh(self, L: int):
        dt = torch.exp(self.log_delta) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        
        dtA = A * dt
        t = torch.arange(L, device=A.device)
        # ZOH initialization:
        log_A_bar = dtA.unsqueeze(-1) * t
        
        A_bar = torch.exp(log_A_bar)
        # The actual formulation:
        # dtB = self.B * dt.unsqueeze(-1)
        # B_bar = dtB * (torch.exp(dtA)-1.) / dtA
        # As dt cancels out we can use:
        # B_bar = self.B * (torch.exp(dtA)-1.) / A
        # For the vandermonde multiplication we use
        # B_bar * C = (self.B * (torch.exp(dtA)-1.) / A) * C = ((self.B*C) * (torch.exp(dtA)-1.) / A)
        # We can define an equivalent parameter: C_bar = (self.B*C) -> ((C_bar) * (torch.exp(dtA)-1.) / A)
        # In total we can substitute for: 
        B_bar_C = C * (torch.exp(dtA) - 1) / A

        K = 2 * torch.einsum('hn, hnl -> hl', B_bar_C, A_bar).real

        del C, A, dt, dtA, log_A_bar, A_bar, B_bar_C

        return K
    

    def kernel_lin(self, L: int):
        delta = torch.exp(self.log_delta)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dA = (1 + delta * A / 2) / (1 - delta * A / 2)
        dB = delta * self.B / (1 - delta * A / 2)
        K = 2 * torch.einsum('hn,hnl->hl', dB * torch.view_as_complex(self.C), (dA.unsqueeze(-1) ** torch.arange(L, device=dA.device))).real

        del delta, A, dA

        return K

        
    def forward(self, u: Tensor): # Shape of u: (batch_size, L, H)
        L = u.shape[-2]
        # u = u.squeeze(0)
        u = u.transpose(-2,-1)
        if self.init_method == 'zoh':
            K = self.kernel_zoh(L)
        elif self.init_method == 'lin':
            K = self.kernel_lin(L)
        else:
            raise ValueError('Invalid initialization method')

        K_f = fft.rfft(K, n=2*L)
        u_f = fft.rfft(u, n=2*L)

        y = fft.irfft(K_f * u_f, n=2*L)[..., :L]
        y = y + u * self.D.unsqueeze(-1)
        y = y.transpose(-2,-1)

        del K, K_f, u_f

        return y
    
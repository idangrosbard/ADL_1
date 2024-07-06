import torch
from torch import Tensor, nn
import numpy as np
from torch.fft import ifft


def calc_kappa(omega, C, Q, _lambda, B, P, delta):
    # Following the algorithm from the paper:
    # CQ = torch.cat([C, Q], dim=1).conj().t()
    middle_part = 2/delta * (1 - omega) / (1 + omega) - _lambda
    middle_part = middle_part.inverse()
    # BP = torch.cat([B, P], dim=1)

    kappa_00 = C.conj().t() @ middle_part @ B
    kappa_01 = C.conj().t() @ middle_part @ P
    kappa_10 = Q.conj().t() @ middle_part @ B
    kappa_11 = Q.conj().t() @ middle_part @ P
    
    return (kappa_00, kappa_01, kappa_10, kappa_11)


def K_hat(omega, kappa):
    return (2/(1 + omega)) * (kappa[0] - kappa[1]*(1+kappa[3]).inverse() @ kappa[2])


def K_hat(C, Q, _lambda, B, P, delta, L):
    ks = torch.linspace(0, 1, L)
    omegas = torch.exp(2j * ks * np.pi)
    kappas = [calc_kappa(omega, C, Q, _lambda, B, P, delta) for omega in omegas]
    K_hats = [K_hat(omega, kappa) for omega, kappa in zip(omegas, kappas)]
    return K_hats


class S4Layer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.delta = Tensor(1, dtype=torch.float32)
        self._lambda = Tensor(hidden_dim, dtype=torch.complex32)
        self.P = Tensor(hidden_dim, dtype=torch.complex32)
        self.Q = Tensor(hidden_dim, dtype=torch.complex32)
        self.B = Tensor(hidden_dim, dtype=torch.complex32)
        self.C = Tensor(hidden_dim, dtype=torch.complex32)
        self.D = Tensor(hidden_dim, dtype=torch.complex32)
    
    def K(self, L):
        # Following the algorithm from the paper:
        A = self._lambda - self.P @ self.Q.conj().t()
        # TODO: calc A_bar, C_bar
        A_bar = (np.eye(A.shape[0]) - self.delta/2 * A).inverse() @ (np.eye(A.shape[0]) + self.delta/2 * A)
        # B_bar = (np.eye(A.shape[0]) - self.delta/2 * A).inverse() @ (self.delta*self.B)
        C_bar = self.C
        C_tilda = (np.eye(A.shape[0]) - A_bar**L).conj().t() @ C_bar
        K_bar = ifft(K_hat(C_tilda, self.Q, self._lambda, self.B, self.P, self.delta, L), n=L)
        return K_bar
    
    def forward(self, u):
        return self.K(len(u)) @ u + self.D * u

"""Shared vector operations for Lie groups and Lie algebras."""

import torch


def sincu(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized sinc function: sin(x) / x"""
    return torch.special.sinc(x / x.new_tensor(torch.pi))


def versine_over_x(x: torch.Tensor) -> torch.Tensor:
    """Computes the versine function divided by x:
    versine(x) = 1 - cos(x)
    versine_over_x(x) = (1 - cos(x)) / x
    """
    return x.new_tensor(0.5) * x * torch.square(sincu(x / x.new_tensor(2.0)))



def sinc_taylor(phi, eps=1e-4):
    # phi: (...,) tensor
    small = phi.abs() < eps
    out   = torch.empty_like(phi)
    # far from zero: use standard
    out[~small] = torch.sin(phi[~small]) / phi[~small]
    # near zero: 1 - φ²/6 + φ⁴/120
    x = phi[small]
    out[small] = 1 - x**2/6 + x**4/120
    return out

def versine_over_x_taylor(phi, eps=1e-4):
    small = phi.abs() < eps
    out   = torch.empty_like(phi)
    out[~small] = (1 - torch.cos(phi[~small])) / phi[~small]
    x = phi[small]
    out[small] = x/2 - x**3/24 + x**5/720
    return out

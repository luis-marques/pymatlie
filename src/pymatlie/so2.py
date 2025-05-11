"""SO(2) Lie group implementation."""

import torch
from pymatlie.base_group import MatrixLieGroup
from pymatlie.vecops import sincu, versine_over_x


class SO2(MatrixLieGroup):
    """SO(2) Lie group implementation (batch-only)."""

    g_dim: int = 1
    matrix_size: tuple = (2, 2)

    @staticmethod
    def vee(tau_hat: torch.Tensor) -> torch.Tensor:
        """Converts Lie algebra matrices (N, 2, 2) to vectors (N, 1)."""
        assert tau_hat.ndim == 3 and tau_hat.shape[-2:] == SO2.matrix_size, "SO(2) vee operator requires a Nx2x2 matrix"
        return tau_hat[..., 1, 0].unsqueeze(-1)

    @staticmethod
    def hat(tau: torch.Tensor) -> torch.Tensor:
        """Converts vectors (N, 1) to Lie algebra matrices (N, 2, 2)."""
        assert tau.ndim == 2 and tau.shape[-1] == SO2.g_dim, f"hat requires shape (N, 1), got {tau.shape}"
        hat_matrix = torch.zeros((tau.shape[0], 2, 2), device=tau.device, dtype=tau.dtype)  # (N, 2, 2)
        hat_matrix[..., 0, 1] = -tau[..., 0]
        hat_matrix[..., 1, 0] = tau[..., 0]
        return hat_matrix

    @staticmethod
    def exp(tau: torch.Tensor) -> torch.Tensor:
        """Computes the matrix exponential of an SO(2) Lie algebra element."""
        assert tau.ndim == 2 and tau.shape[-1] == SO2.g_dim, "SO(2) exp requires a Nx2x2 matrix"
        sin = torch.sin(tau[..., 0])
        cos = torch.cos(tau[..., 0])
        g = torch.zeros((tau.shape[0], 2, 2), device=tau.device)  # (N, 2, 2)
        g[..., 0, 0] = cos
        g[..., 0, 1] = -sin
        g[..., 1, 0] = sin
        g[..., 1, 1] = cos
        return g

    @staticmethod
    def skew_sym() -> torch.Tensor:
        """Returns the 2×2 constant skew-symmetric matrix for SO(2): [[0, -1],
        [1, 0]]"""
        return torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)  # (2, 2)

    @staticmethod
    def logm(g: torch.Tensor) -> torch.Tensor:
        """Computes the matrix logarithm of an SO(2) Lie group element."""
        assert g.ndim == 3 and g.shape[-2:] == SO2.matrix_size, "SO(2) logm requires a Nx2x2 matrix"
        angle = torch.atan2(g[..., 1, 0], g[..., 0, 0])
        hat_matrix = torch.zeros((g.shape[0], 2, 2), device=g.device)
        hat_matrix[..., 0, 1] = -angle
        hat_matrix[..., 1, 0] = angle
        return hat_matrix

    @staticmethod
    def log(g: torch.Tensor) -> torch.Tensor:
        """Computes the matrix logarithm of an SO(2) Lie group element."""
        assert g.ndim == 3 and g.shape[-2:] == SO2.matrix_size, "SO(2) log requires a Nx2x2 matrix"
        return torch.atan2(g[..., 1, 0], g[..., 0, 0]).unsqueeze(-1)

    @staticmethod
    def left_jacobian(tau: torch.Tensor) -> torch.Tensor:
        """Computes the left Jacobian J_left of SO(2) for (N, 1) vectors."""
        assert tau.ndim == 2 and tau.shape[-1] == SO2.g_dim, "left_jacobian requires shape (N, 1)"
        tau = tau[..., 0]  # (N,)
        I = torch.eye(2, dtype=tau.dtype, device=tau.device)  # (2, 2)
        skew = SO2.skew_sym().to(tau.device).to(tau.dtype)  # (2, 2)

        return sincu(tau)[..., None, None] * I + versine_over_x(tau)[..., None, None] * skew

    @staticmethod
    def left_jacobian_inverse(tau: torch.Tensor) -> torch.Tensor:
        """Inverse of left Jacobian of SO(2), input shape (N, 1), output (N, 2,
        2)."""
        assert tau.ndim == 2 and tau.shape[-1] == SO2.g_dim, "left_jacobian_inverse requires shape (N, 1)"
        tau = tau[..., 0]
        A = sincu(tau)  # (N,)
        B = versine_over_x(tau)  # (N,)
        denom = A**2 + B**2  # (N,)

        row0 = torch.stack([A, B], dim=-1)  # (N, 2)
        row1 = torch.stack([-B, A], dim=-1)  # (N, 2)
        return torch.stack([row0, row1], dim=-2) / denom[..., None, None]

    @staticmethod
    def adjoint_matrix(g: torch.Tensor) -> torch.Tensor:
        """Computes the adjoint matrix of an SO(2) element."""
        assert g.ndim == 3 and g.shape[-2:] == SO2.matrix_size, "adjoint_matrix requires shape (N, 2, 2)"
        return torch.eye(2, dtype=g.dtype, device=g.device).repeat(g.shape[0], 1, 1)

    @staticmethod
    def g_to_euclidean_vector(g: torch.Tensor) -> torch.Tensor:
        """Converts a Lie group element (SO(2) matrix) to a Lie algebra
        element."""
        return SO2.log(g)

    @staticmethod
    def euclidean_vector_to_g(x):
        """Converts a Lie algebra element (SO(2) vector) to a Lie group
        element."""
        return SO2.exp(x)

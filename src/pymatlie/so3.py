"""SO(3) Lie group implementation."""

from dataclasses import field
from typing import Tuple

import torch

from pymatlie.base_group import MatrixLieGroup
from pymatlie.quaternions import matrix_to_quaternion, quaternion_to_matrix


class SO3(MatrixLieGroup):
    """SO(3) Lie group implementation (batch-only)."""

    g_dim: int = 3
    matrix_size: tuple = (3, 3)
    g_names: Tuple[str, ...] = field(
        default=(
            r"$r_{11}$",
            r"$r_{12}$",
            r"$r_{13}$",
            r"$x$",
            r"$r_{21}$",
            r"$r_{22}$",
            r"$r_{23}$",
            r"$y$",
            r"$r_{31}$",
            r"$r_{32}$",
            r"$r_{33}$",
            r"$z$",
            r"$0$",
            r"$0$",
            r"$0$",
            r"$1$",
        ),
        init=False,
    )
    xi_names: Tuple[str, str, str, str, str, str] = field(
        default=(r"$v_x$", r"$v_y$", r"$v_z$", r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"), init=False
    )

    @staticmethod
    def vee(tau_hat: torch.Tensor) -> torch.Tensor:
        """Converts rotation matrices (N, 3, 3) to quaternions (N, 4) [qx, qy, qz, qw]."""
        assert tau_hat.ndim == 3 and tau_hat.shape[-2:] == SO3.matrix_size, "SO(3) vee operator requires a Nx3x3 matrix"
        return matrix_to_quaternion(tau_hat)

    @staticmethod
    def inverse(g: torch.Tensor) -> torch.Tensor:
        """Inverse of SO(3) Lie group element."""
        assert g.ndim == 3 and g.shape[-2:] == SO3.matrix_size, "SO(3) inverse requires a Nx3x3 matrix"
        return g.transpose(-2, -1)

    @staticmethod
    def hat(tau: torch.Tensor) -> torch.Tensor:
        """Converts vectors (N, 3) to Lie algebra matrices (N, 3, 3)."""
        assert tau.ndim == 2 and tau.shape[-1] == SO3.g_dim, f"hat requires shape (N, 3), got {tau.shape}"
        hat_matrix = torch.zeros((tau.shape[0], 3, 3), device=tau.device, dtype=tau.dtype)
        hat_matrix[..., 0, 1] = -tau[..., 2]
        hat_matrix[..., 0, 2] = tau[..., 1]
        hat_matrix[..., 1, 0] = tau[..., 2]
        hat_matrix[..., 1, 2] = -tau[..., 0]
        hat_matrix[..., 2, 0] = -tau[..., 1]
        hat_matrix[..., 2, 1] = tau[..., 0]
        return hat_matrix

    @staticmethod
    def logm(g: torch.Tensor) -> torch.Tensor:
        """Computes the matrix logarithm of an SO(3) Lie group element."""
        assert g.ndim == 3 and g.shape[-2:] == SO3.matrix_size, "SO(3) logm requires a Nx3x3 matrix"

        # Compute the angle
        trace = torch.diagonal(g, dim1=-2, dim2=-1).sum(dim=-1)  # (N,)
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))  # (N,)

        # Handle small angles
        small_angle = angle < 1e-4

        hat_matrix = torch.zeros_like(g)

        if not small_angle.all():
            # Standard case
            sin_angle = torch.sin(angle)
            factor = angle / (2 * sin_angle)

            # Skew-symmetric part
            skew = factor.unsqueeze(-1).unsqueeze(-1) * (g - g.transpose(-2, -1))
            hat_matrix[~small_angle] = skew[~small_angle]

        # Small angle case: use first-order approximation
        if small_angle.any():
            hat_matrix[small_angle] = 0.5 * (g[small_angle] - g[small_angle].transpose(-2, -1))

        return hat_matrix

    @staticmethod
    def map_configuration_to_q(g: torch.Tensor) -> torch.Tensor:
        """Converts a rotation matrix to quaternion representation."""
        return matrix_to_quaternion(g)

    @staticmethod
    def map_q_to_configuration(x: torch.Tensor) -> torch.Tensor:
        """Converts a Lie algebra element (SO(3) vector) to a Lie group element."""
        return quaternion_to_matrix(x)

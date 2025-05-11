"""SE(2) group implementation."""

from dataclasses import dataclass, field
from typing import Tuple

import torch
from pymatlie.base_group import MatrixLieGroup
from pymatlie.so2 import SO2


@dataclass(frozen=True)
class SE2(MatrixLieGroup):
    """Special Euclidean group SE(2)."""

    inertia_matrix: torch.Tensor
    g_dim: int = 3
    matrix_size: tuple = (3, 3)
    B: torch.Tensor = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    u_dim: int = 2
    g_names: Tuple[str, ...] = field(
        default=(r"$c_\theta$", r"-s_\theta$", r"$x$", r"$s_\theta$", r"c_\theta$", r"$y$", r"0$", r"0$", r"1$"), init=False
    )
    xi_names: Tuple[str, str, str] = field(default=(r"v_x", r"v_y", r"\omega"), init=False)

    @staticmethod
    def vee(tau_hat: torch.Tensor) -> torch.Tensor:
        """Converts a Lie algebra element (3x3 matrix) to a 3D vector."""
        assert tau_hat.ndim == 3 and tau_hat.shape[-2:] == SE2.matrix_size, f"vee requires shape (N, {SE2.matrix_size}), got {tau_hat.shape}"
        theta = SO2.vee(tau_hat[..., :2, :2])
        return torch.cat([tau_hat[..., :2, 2], theta], dim=-1)  # (N, 3)

    @staticmethod
    def hat(tau: torch.Tensor) -> torch.Tensor:
        """Converts a 3D vector into a Lie algebra matrix (SE(2) hat
        operator)"""
        assert tau.ndim == 2 and tau.shape[-1] == SE2.g_dim, f"hat requires shape (N, {SE2.g_dim}), got {tau.shape}"
        hat_matrix = torch.zeros((tau.shape[0], 3, 3), device=tau.device, dtype=tau.dtype)
        hat_matrix[..., :2, :2] = SO2.hat(tau[..., 2:3])
        hat_matrix[..., :2, 2] = tau[..., :2]
        return hat_matrix

    @staticmethod
    def logm(g: torch.Tensor) -> torch.Tensor:
        """Computes the matrix logarithm of an SE(2) Lie group element."""
        assert g.ndim == 3 and g.shape[-2:] == SE2.matrix_size, f"logm requires shape (N, {SE2.matrix_size}), got {g.shape}"
        theta = SO2.log(g[..., :2, :2])  # SO(2) logarithm
        t = g[..., :2, 2].unsqueeze(-1)  # Extract translation

        rho = SO2.left_jacobian_inverse(theta) @ t  # Transform translation using the inverse of the left Jacobian

        tau_hat = torch.zeros((*g.shape[:-2], 3, 3), device=g.device, dtype=g.dtype)
        tau_hat[..., :2, :2] = SO2.hat(theta)  # Set the rotation part
        tau_hat[..., :2, 2] = rho.squeeze(-1)  # Set the translation part
        return tau_hat

    # # @torch.compile
    @staticmethod
    def exp(tau: torch.Tensor) -> torch.Tensor:
        """Computes the exponential map of SE(2)"""
        assert tau.ndim == 2 and tau.shape[-1] == SE2.g_dim, f"exp requires shape (N, {SE2.g_dim}), got {tau.shape}"
        rho = tau[..., :2]
        phi = tau[..., 2:3]

        g = torch.eye(3, device=tau.device).repeat(*tau.shape[:-1], 1, 1)
        g[..., :2, :2] = SO2.exp(phi)  # SO(2) exponential, returns (N, 2, 2)

        V = SO2.left_jacobian(phi)  # (N, 2, 2)
        g[..., :2, 2] = torch.matmul(V, rho.unsqueeze(-1))[..., 0]  # (N, 2)
        return g

    # @torch.compile
    @staticmethod
    def log(g: torch.Tensor) -> torch.Tensor:
        """Computes the logarithm map of SE(2)"""
        assert g.ndim == 3 and g.shape[-2:] == SE2.matrix_size, f"log requires shape (N, {SE2.matrix_size}), got {g.shape}"
        phi = SO2.log(g[..., :2, :2])
        t = g[..., :2, 2]
        V_inv = SO2.left_jacobian_inverse(phi)
        rho = torch.matmul(V_inv, t.unsqueeze(-1)).squeeze(-1)
        return torch.cat([rho, phi], dim=-1)

    @classmethod
    def left_jacobian(cls, tau: torch.Tensor) -> torch.Tensor:
        """Computes the left Jacobian of SE(2)."""
        assert tau.ndim == 2 and tau.shape[-1] == cls.g_dim, f"left_jacobian requires shape (N, {cls.g_dim}), got {tau.shape}"
        theta = tau[..., 2]
        x, y = tau[..., 0], tau[..., 1]

        jac = torch.eye(3, dtype=tau.dtype, device=tau.device).repeat(tau.shape[0], 1, 1)
        jac[..., :2, :2] = SO2.left_jacobian(theta.unsqueeze(-1))  # (N, 2, 2)
        jac[..., 0, 2] = (theta * x + y - y * torch.cos(theta) - x * torch.sin(theta)) / (theta**2 + 1e-15)
        jac[..., 1, 2] = (theta * y - x + x * torch.cos(theta) - y * torch.sin(theta)) / (theta**2 + 1e-15)
        return jac

    @staticmethod
    def adjoint_matrix(g: torch.Tensor) -> torch.Tensor:
        """Computes the adjoint matrix of SE(2)"""
        assert g.ndim == 3 and g.shape[-2:] == SE2.matrix_size, f"adjoint_matrix requires shape (N, {SE2.matrix_size}), got {g.shape}"
        adj = torch.zeros_like(g)
        adj[..., :2, :2] = g[..., :2, :2]
        adj[..., :2, 2] = torch.stack([g[..., 1, 2], -g[..., 0, 2]], dim=-1)
        adj[..., 2, 2] = 1
        return adj

    @staticmethod
    def ad_operator(xi: torch.Tensor) -> torch.Tensor:
        """Infinitesimal adjoint operator of SE(2)"""
        assert xi.ndim == 2 and xi.shape[-1] == SE2.g_dim, f"ad_operator requires shape (N, {SE2.g_dim}), got {xi.shape}"
        hat_matrix = torch.zeros((*xi.shape[:-1], 3, 3), device=xi.device, dtype=xi.dtype)
        hat_matrix[..., :2, :2] = SO2.hat(xi[..., 2:3])
        hat_matrix[..., 0, 2] = xi[..., 1]
        hat_matrix[..., 1, 2] = -xi[..., 0]
        return hat_matrix

    @staticmethod
    def coadjoint_operator(xi: torch.Tensor) -> torch.Tensor:
        """Infinitesimal coadjoint operator of SE(2)"""
        assert xi.ndim == 2 and xi.shape[-1] == SE2.g_dim, f"coadjoint_operator requires shape (N, {SE2.g_dim}), got {xi.shape}"
        return SE2.ad_operator(xi).transpose(-2, -1)

    @staticmethod
    def compute_twist_map(q: torch.Tensor) -> torch.Tensor:
        """Compute body Jacobian for unicycle."""
        theta = q[:, 2]
        J = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
        J[:, 0, 0] = torch.cos(theta)
        J[:, 0, 1] = torch.sin(theta)
        J[:, 1, 0] = -torch.sin(theta)
        J[:, 1, 1] = torch.cos(theta)
        J[:, 2, 2] = 1.0
        return J

    @staticmethod
    def compute_twist_map_inverse(q: torch.Tensor) -> torch.Tensor:
        """Compute J⁻¹(q)=J(q)^T mapping ξ→q̇ without any pinv."""
        J = SE2.compute_twist_map(q)
        return J.transpose(-2, -1)
    
    @staticmethod
    def map_q_to_configuration(q: torch.Tensor) -> torch.Tensor:
        assert q.ndim == 2 and q.shape[-1] == SE2.g_dim, f"euclidean_vector_to_g requires shape (N, {SE2.g_dim}), got {q.shape}"
        g = torch.eye(3, device=q.device).repeat(*q.shape[:-1], 1, 1)
        g[..., 0:2, 2] = q[..., :2]
        g[..., :2, :2] = SO2.exp(q[..., 2].unsqueeze(-1))
        return g
    
    @staticmethod
    def map_configuration_to_q(g: torch.Tensor) -> torch.Tensor:
        assert g.ndim == 3 and g.shape[-2:] == SE2.matrix_size, f"g_to_euclidean_vector requires shape (N, {SE2.matrix_size}), got {g.shape}"
        theta = torch.atan2(g[..., 1, 0], g[..., 0, 0])
        x = g[..., 0, 2]  # (N,)
        y = g[..., 1, 2]  # (N,)
        return torch.stack((x, y, theta), dim=1)  # (N, 3)

    @staticmethod
    def map_dq_to_velocity(q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        assert q.ndim == 2 and q.shape[1] == SE2.g_dim, f"map_dq_to_SE2velocity requires shape (N, {SE2.g_dim}), got {q.shape}"
        assert dq.ndim == 2 and dq.shape[1] == SE2.g_dim, f"dq_to_SE2velocity requires shape (N, {SE2.g_dim}), got {dq.shape}"
        J_twist = SE2.compute_twist_map(q)
        xi = torch.bmm(J_twist, dq.unsqueeze(-1)).squeeze(-1)  # (N, g_dim)
        # assert xi.shape == (dq.shape[0], self.g_dim), f"xi shape mismatch: expected {(dq.shape[0], self.g_dim)}, got {xi.shape}"
        return xi
    
    @staticmethod
    def map_velocity_to_dq(q: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        assert q.ndim == 2 and q.shape[1] == SE2.g_dim, f"map_velocity_to_dq requires shape (N, {SE2.g_dim}), got {q.shape}"
        assert vel.ndim == 2 and vel.shape[1] == SE2.g_dim, f"velocity_to_dq requires shape (N, {SE2.g_dim}), got {vel.shape}"
        J_twist_inv = SE2.compute_twist_map_inverse(q)
        dq = torch.bmm(J_twist_inv, vel.unsqueeze(-1)).squeeze(-1)
        # assert dq.shape == (vel.shape[0], self.g_dim), f"dq shape mismatch: expected {(N, self.g_dim)}, got {dq.shape}"
        return dq
"""SE(3) group implementation."""

from dataclasses import dataclass, field
from typing import Tuple

import torch
from pymatlie.base_group import MatrixLieGroup
from pymatlie.so3 import SO3


@dataclass(frozen=True)
class SE3(MatrixLieGroup):
    """Special Euclidean group SE(3)."""

    inertia_matrix: torch.Tensor
    g_dim: int = 6
    matrix_size: tuple = (4, 4)
    g_names: Tuple[str, ...] = field(
        default=(r"$r_{11}$", r"$r_{12}$", r"$r_{13}$", r"$x$",
                r"$r_{21}$", r"$r_{22}$", r"$r_{23}$", r"$y$",
                r"$r_{31}$", r"$r_{32}$", r"$r_{33}$", r"$z$",
                r"$0$", r"$0$", r"$0$", r"$1$"), init=False
    )
    xi_names: Tuple[str, str, str, str, str, str] = field(
        default=(r"$v_x$", r"$v_y$", r"$v_z$", r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"), init=False
    )

    @staticmethod
    def vee(tau_hat: torch.Tensor) -> torch.Tensor:
        """Converts a Lie algebra element (4x4 matrix) to a 6D vector."""
        assert tau_hat.ndim == 3 and tau_hat.shape[-2:] == SE3.matrix_size, f"vee requires shape (N, {SE3.matrix_size}), got {tau_hat.shape}"
        rho = tau_hat[..., :3, 3]  # translation part (N, 3)
        omega = SO3.vee(tau_hat[..., :3, :3])  # rotation part (N, 3)
        return torch.cat([rho, omega], dim=-1)  # (N, 6)

    @staticmethod
    def hat(tau: torch.Tensor) -> torch.Tensor:
        """Converts a 6D vector into a Lie algebra matrix (SE(3) hat operator)."""
        assert tau.ndim == 2 and tau.shape[-1] == SE3.g_dim, f"hat requires shape (N, {SE3.g_dim}), got {tau.shape}"
        hat_matrix = torch.zeros((tau.shape[0], 4, 4), device=tau.device, dtype=tau.dtype)
        hat_matrix[..., :3, :3] = SO3.hat(tau[..., 3:6])  # rotation part
        hat_matrix[..., :3, 3] = tau[..., :3]  # translation part
        return hat_matrix

    @staticmethod
    def logm(g: torch.Tensor) -> torch.Tensor:
        """Computes the matrix logarithm of an SE(3) Lie group element."""
        assert g.ndim == 3 and g.shape[-2:] == SE3.matrix_size, f"logm requires shape (N, {SE3.matrix_size}), got {g.shape}"
        
        R = g[..., :3, :3]  # rotation part (N, 3, 3)
        t = g[..., :3, 3].unsqueeze(-1)  # translation part (N, 3, 1)
        
        # Get rotation part
        omega = SO3.log(R)  # (N, 3)
        
        # Compute V^{-1}
        V_inv = SO3.left_jacobian_inverse(omega)  # (N, 3, 3)
        
        # Transform translation
        rho = torch.bmm(V_inv, t).squeeze(-1)  # (N, 3)
        
        tau_hat = torch.zeros((*g.shape[:-2], 4, 4), device=g.device, dtype=g.dtype)
        tau_hat[..., :3, :3] = SO3.hat(omega)  # Set the rotation part
        tau_hat[..., :3, 3] = rho  # Set the translation part
        return tau_hat

    @staticmethod
    def exp(tau: torch.Tensor) -> torch.Tensor:
        """Computes the exponential map of SE(3)."""
        assert tau.ndim == 2 and tau.shape[-1] == SE3.g_dim, f"exp requires shape (N, {SE3.g_dim}), got {tau.shape}"
        rho = tau[..., :3]  # translation part (N, 3)
        omega = tau[..., 3:6]  # rotation part (N, 3)

        g = torch.eye(4, device=tau.device, dtype=tau.dtype).repeat(*tau.shape[:-1], 1, 1)
        g[..., :3, :3] = SO3.exp(omega)  # SO(3) exponential (N, 3, 3)

        V = SO3.left_jacobian(omega)  # (N, 3, 3)
        g[..., :3, 3] = torch.bmm(V, rho.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        return g

    @staticmethod
    def log(g: torch.Tensor) -> torch.Tensor:
        """Computes the logarithm map of SE(3)."""
        assert g.ndim == 3 and g.shape[-2:] == SE3.matrix_size, f"log requires shape (N, {SE3.matrix_size}), got {g.shape}"
        R = g[..., :3, :3]  # rotation part
        t = g[..., :3, 3]  # translation part
        
        omega = SO3.log(R)  # (N, 3)
        V_inv = SO3.left_jacobian_inverse(omega)  # (N, 3, 3)
        rho = torch.bmm(V_inv, t.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        
        return torch.cat([rho, omega], dim=-1)  # (N, 6)

    @classmethod
    def left_jacobian(cls, tau: torch.Tensor) -> torch.Tensor:
        """Computes the left Jacobian of SE(3)."""
        assert tau.ndim == 2 and tau.shape[-1] == cls.g_dim, f"left_jacobian requires shape (N, {cls.g_dim}), got {tau.shape}"
        
        rho = tau[..., :3]  # translation part (N, 3)
        omega = tau[..., 3:6]  # rotation part (N, 3)
        
        # Initialize Jacobian
        jac = torch.eye(6, dtype=tau.dtype, device=tau.device).repeat(tau.shape[0], 1, 1)
        
        # Rotation part
        J_rot = SO3.left_jacobian(omega)  # (N, 3, 3)
        jac[..., 3:6, 3:6] = J_rot
        
        # Translation part  
        jac[..., :3, :3] = J_rot
        
        # Cross-coupling term
        angle = torch.norm(omega, dim=-1)  # (N,)
        small_angle = angle < 1e-4
        
        # For small angles, use approximation
        if small_angle.all():
            Q = 0.5 * SO3.hat(rho)
        else:
            # Full formula for cross-coupling
            sin_angle = torch.sin(angle)
            cos_angle = torch.cos(angle)
            
            # Coefficients
            A = (1 - cos_angle) / (angle ** 2)
            B = (angle - sin_angle) / (angle ** 3)
            
            # Handle small angles separately
            A_safe = torch.where(small_angle, 0.5 * torch.ones_like(A), A)
            B_safe = torch.where(small_angle, 1/6 * torch.ones_like(B), B)
            
            # Compute Q matrix
            rho_hat = SO3.hat(rho)
            omega_hat = SO3.hat(omega)
            
            Q = A_safe.unsqueeze(-1).unsqueeze(-1) * rho_hat + \
                B_safe.unsqueeze(-1).unsqueeze(-1) * torch.bmm(omega_hat, rho_hat)
        
        jac[..., :3, 3:6] = Q
        
        return jac

    @classmethod
    def left_jacobian_inverse(cls, tau: torch.Tensor) -> torch.Tensor:
        """Analytic inverse of the left Jacobian for SE(3)."""
        assert tau.ndim == 2 and tau.shape[-1] == cls.g_dim, f"left_jacobian_inverse requires shape (N, {cls.g_dim}), got {tau.shape}"
        
        rho = tau[..., :3]  # translation part (N, 3)
        omega = tau[..., 3:6]  # rotation part (N, 3)
        
        # Initialize inverse Jacobian
        jac_inv = torch.eye(6, dtype=tau.dtype, device=tau.device).repeat(tau.shape[0], 1, 1)
        
        # Rotation part
        J_rot_inv = SO3.left_jacobian_inverse(omega)  # (N, 3, 3)
        jac_inv[..., 3:6, 3:6] = J_rot_inv
        
        # Translation part
        jac_inv[..., :3, :3] = J_rot_inv
        
        # Cross-coupling term (negative of the forward case)
        angle = torch.norm(omega, dim=-1)  # (N,)
        small_angle = angle < 1e-4
        
        if small_angle.all():
            Q_inv = -0.5 * SO3.hat(rho)
        else:
            # Full formula for cross-coupling inverse
            sin_angle = torch.sin(angle)
            cos_angle = torch.cos(angle)
            
            # Coefficients for inverse
            A_inv = (1 - cos_angle) / (angle ** 2)
            B_inv = (angle - sin_angle) / (angle ** 3)
            
            # Handle small angles
            A_inv_safe = torch.where(small_angle, 0.5 * torch.ones_like(A_inv), A_inv)
            B_inv_safe = torch.where(small_angle, 1/6 * torch.ones_like(B_inv), B_inv)
            
            # Compute Q_inv matrix
            rho_hat = SO3.hat(rho)
            omega_hat = SO3.hat(omega)
            
            Q_inv = -A_inv_safe.unsqueeze(-1).unsqueeze(-1) * rho_hat - \
                    B_inv_safe.unsqueeze(-1).unsqueeze(-1) * torch.bmm(omega_hat, rho_hat)
        
        jac_inv[..., :3, 3:6] = torch.bmm(J_rot_inv, Q_inv)
        
        return jac_inv

    @staticmethod
    def adjoint_matrix(g: torch.Tensor) -> torch.Tensor:
        """Computes the adjoint matrix of SE(3)."""
        assert g.ndim == 3 and g.shape[-2:] == SE3.matrix_size, f"adjoint_matrix requires shape (N, {SE3.matrix_size}), got {g.shape}"
        
        R = g[..., :3, :3]  # rotation part (N, 3, 3)
        t = g[..., :3, 3]  # translation part (N, 3)
        
        adj = torch.zeros((g.shape[0], 6, 6), device=g.device, dtype=g.dtype)
        adj[..., :3, :3] = R
        adj[..., 3:6, 3:6] = R
        adj[..., :3, 3:6] = SO3.hat(t) @ R
        
        return adj

    @staticmethod
    def ad_operator(xi: torch.Tensor) -> torch.Tensor:
        """Infinitesimal adjoint operator of SE(3)."""
        assert xi.ndim == 2 and xi.shape[-1] == SE3.g_dim, f"ad_operator requires shape (N, {SE3.g_dim}), got {xi.shape}"
        
        rho = xi[..., :3]  # translation part (N, 3)
        omega = xi[..., 3:6]  # rotation part (N, 3)
        
        ad_matrix = torch.zeros((xi.shape[0], 6, 6), device=xi.device, dtype=xi.dtype)
        ad_matrix[..., :3, :3] = SO3.hat(omega)
        ad_matrix[..., 3:6, 3:6] = SO3.hat(omega)
        ad_matrix[..., :3, 3:6] = SO3.hat(rho)
        
        return ad_matrix

    @staticmethod
    def coadjoint_operator(xi: torch.Tensor) -> torch.Tensor:
        """Infinitesimal coadjoint operator of SE(3)."""
        assert xi.ndim == 2 and xi.shape[-1] == SE3.g_dim, f"coadjoint_operator requires shape (N, {SE3.g_dim}), got {xi.shape}"
        return SE3.ad_operator(xi).transpose(-2, -1)

    @staticmethod
    def map_q_to_configuration(q: torch.Tensor) -> torch.Tensor:
        """Map configuration vector (x, y, z, q_x, q_y, q_z, q_w) to SE(3) matrix."""
        assert q.ndim == 2 and q.shape[-1] == 7, f"map_q_to_configuration requires shape (N, 7), got {q.shape}"
        
        g = torch.eye(4, device=q.device, dtype=q.dtype).repeat(*q.shape[:-1], 1, 1)
        g[..., :3, 3] = q[..., :3]  # translation
        g[..., :3, :3] = SO3.map_q_to_configuration(q[..., 3:7])  # rotation from euler angles
        return g
    
    @staticmethod
    def map_configuration_to_q(g: torch.Tensor) -> torch.Tensor:
        """Map SE(3) matrix to configuration vector (x, y, z, q_x, q_y, q_z, q_w)."""
        assert g.ndim == 3 and g.shape[-2:] == SE3.matrix_size, f"map_configuration_to_q requires shape (N, {SE3.matrix_size}), got {g.shape}"
        
        t = g[..., :3, 3]  # translation (N, 3)
        R = g[..., :3, :3]  # rotation (N, 3, 3)
        # convert to quaternion
        q  = SO3.matrix_to_quaternion(R)
        return torch.cat([t, q], dim=-1)

    @staticmethod
    def se3_to_se2(g_se3: torch.Tensor, ignore_axis: str) -> torch.Tensor:
        """Convert SE(3) poses to SE(2) by projecting to a 2D plane.
        
        Args:
            g_se3: SE(3) group elements of shape (N, 4, 4)
            ignore_axis: Which axis to ignore ('x', 'y', or 'z')
        
        Returns:
            SE(2) group elements of shape (N, 3, 3)
        """
        assert g_se3.ndim == 3 and g_se3.shape[-2:] == SE3.matrix_size, f"se3_to_se2 requires shape (N, 4, 4), got {g_se3.shape}"
        assert ignore_axis in ['x', 'y', 'z'], f"ignore_axis must be 'x', 'y', or 'z', got {ignore_axis}"
        
        g_se2 = torch.eye(3, device=g_se3.device, dtype=g_se3.dtype).repeat(g_se3.shape[0], 1, 1)
        
        if ignore_axis == 'z':
            # Standard case: project onto xy-plane, ignore z
            g_se2[..., :2, :2] = g_se3[..., :2, :2]  # Extract 2x2 rotation
            g_se2[..., :2, 2] = g_se3[..., :2, 3]   # Extract x,y translation
        elif ignore_axis == 'y':
            # Project onto xz-plane, ignore y
            g_se2[..., 0, 0] = g_se3[..., 0, 0]   # R11
            g_se2[..., 0, 1] = g_se3[..., 0, 2]   # R13  
            g_se2[..., 1, 0] = g_se3[..., 2, 0]   # R31
            g_se2[..., 1, 1] = g_se3[..., 2, 2]   # R33
            g_se2[..., 0, 2] = g_se3[..., 0, 3]   # x translation
            g_se2[..., 1, 2] = g_se3[..., 2, 3]   # z translation
        elif ignore_axis == 'x':
            # Project onto yz-plane, ignore x
            g_se2[..., 0, 0] = g_se3[..., 1, 1]   # R22
            g_se2[..., 0, 1] = g_se3[..., 1, 2]   # R23
            g_se2[..., 1, 0] = g_se3[..., 2, 1]   # R32
            g_se2[..., 1, 1] = g_se3[..., 2, 2]   # R33
            g_se2[..., 0, 2] = g_se3[..., 1, 3]   # y translation
            g_se2[..., 1, 2] = g_se3[..., 2, 3]   # z translation
            
        return g_se2
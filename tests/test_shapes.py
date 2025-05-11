"""Testing output shapes."""

import pytest
import torch
from pymatlie.se2 import SE2
from pymatlie.so2 import SO2

LIE_GROUPS = [SE2, SO2]


@pytest.mark.parametrize("LieGroup", LIE_GROUPS)
def test_size(LieGroup):
    """Test matrix size properties."""
    assert LieGroup.matrix_size[0] == LieGroup.matrix_size[1]
    assert LieGroup.matrix_size[0] * LieGroup.matrix_size[1] == LieGroup.SIZE
    assert LieGroup.SIZE >= LieGroup.g_dim


@pytest.mark.parametrize("LieGroup", LIE_GROUPS)
def test_identity(LieGroup):
    """Test that get_identity() returns a valid identity matrix."""
    identity = LieGroup.get_identity()
    assert identity.shape == LieGroup.matrix_size
    assert torch.allclose(identity, torch.eye(*LieGroup.matrix_size, dtype=identity.dtype, device=identity.device))


@pytest.mark.parametrize("LieGroup", LIE_GROUPS)
def test_shapes(LieGroup):
    """Test shapes of batched operations for LieGroup."""
    N = 10
    mat_shape = (N, *LieGroup.matrix_size)
    vec_shape = (N, LieGroup.g_dim)

    # Sample input
    g = torch.eye(LieGroup.matrix_size[0]).repeat(N, 1, 1)
    tau = torch.randn(vec_shape)

    # ----- algebra-related -----
    tau_hat = LieGroup.hat(tau)
    assert tau_hat.shape == mat_shape, f"hat: expected {mat_shape}, got {tau_hat.shape}"

    tau_recovered = LieGroup.vee(tau_hat)
    assert tau_recovered.shape == vec_shape, f"vee: expected {vec_shape}, got {tau_recovered.shape}"

    # ----- group-related -----
    g_exp = LieGroup.exp(tau)
    assert g_exp.shape == mat_shape, f"exp: expected {mat_shape}, got {g_exp.shape}"

    g_log = LieGroup.log(g)
    assert g_log.shape == vec_shape, f"log: expected {vec_shape}, got {g_log.shape}"

    g_logm = LieGroup.logm(g)
    assert g_logm.shape == mat_shape, f"logm: expected {mat_shape}, got {g_logm.shape}"

    g_expm = LieGroup.expm(tau_hat)
    assert g_expm.shape == mat_shape, f"expm: expected {mat_shape}, got {g_expm.shape}"

    g_from_vec = LieGroup.euclidean_vector_to_g(tau)
    assert g_from_vec.shape == mat_shape, f"euclidean_vector_to_g: expected {mat_shape}, got {g_from_vec.shape}"

    tau_from_g = LieGroup.g_to_euclidean_vector(g)
    assert tau_from_g.shape == vec_shape, f"g_to_euclidean_vector: expected {vec_shape}, got {tau_from_g.shape}"

    # ----- Jacobians -----
    if hasattr(LieGroup, "left_jacobian"):
        J_left = LieGroup.left_jacobian(tau)
        assert J_left.shape == mat_shape, f"left_jacobian: expected {mat_shape}, got {J_left.shape}"

        J_right = LieGroup.right_jacobian(tau)
        assert J_right.shape == mat_shape, f"right_jacobian: expected {mat_shape}, got {J_right.shape}"

    if hasattr(LieGroup, "left_jacobian_inverse"):
        J_left_inv = LieGroup.left_jacobian_inverse(tau)
        assert J_left_inv.shape == mat_shape, f"left_jacobian_inverse: expected {mat_shape}, got {J_left_inv.shape}"

    adj = LieGroup.adjoint_matrix(g)
    assert adj.shape == mat_shape, f"adjoint_matrix: expected {mat_shape}, got {adj.shape}"

    # ----- right_plus / right_minus -----
    g_plus = LieGroup.right_plus(g, tau)
    assert g_plus.shape == mat_shape, f"right_plus: expected {mat_shape}, got {g_plus.shape}"

    g_minus = LieGroup.right_minus(g, g_plus)
    assert g_minus.shape == vec_shape, f"right_minus: expected {vec_shape}, got {g_minus.shape}"

    # ----- invariant errors -----
    E_right = LieGroup.right_invariant_error(g_plus, g)
    assert E_right.shape == mat_shape, f"right_invariant_error: expected {mat_shape}, got {E_right.shape}"

    E_left = LieGroup.left_invariant_error(g_plus, g)
    assert E_left.shape == mat_shape, f"left_invariant_error: expected {mat_shape}, got {E_left.shape}"

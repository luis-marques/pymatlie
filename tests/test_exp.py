"""Test the exponential map for Lie groups."""

import pytest
import torch
from pymatlie.se2 import SE2
from pymatlie.so2 import SO2
from torch.linalg import matrix_exp

LIE_GROUPS = [SE2, SO2]


@pytest.mark.parametrize("LieGroup", LIE_GROUPS)
def test_exp(LieGroup):
    """Test the exponential map for Lie groups."""
    assert LieGroup.matrix_size[0] == LieGroup.matrix_size[1], "Matrix must be square"
    assert LieGroup.matrix_size[0] * LieGroup.matrix_size[1] == LieGroup.SIZE, "Matrix size must match SIZE"
    assert LieGroup.SIZE >= LieGroup.g_dim, "Size must be greater than or equal to dimension"

    psi = torch.randn(200, LieGroup.g_dim)
    psi_hat = LieGroup.hat(psi)

    Psi = LieGroup.expm(psi_hat)

    assert torch.allclose(Psi, matrix_exp(psi_hat), atol=1e-6), "Exponential map failed to match torch.linalg.matrix_exp"

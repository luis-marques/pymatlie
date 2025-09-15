"""Test the exponential map for Lie groups."""

import pytest
import torch
from torch.linalg import matrix_exp

from pymatlie.se2 import SE2
from pymatlie.so2 import SO2

LIE_GROUPS = [SE2, SO2]


@pytest.mark.parametrize("LieGroup", LIE_GROUPS)
def test_exp(LieGroup):
    """Test the exponential map for Lie groups."""
    assert LieGroup.matrix_size[0] == LieGroup.matrix_size[1], "Matrix must be square"

    psi = torch.randn(200, LieGroup.g_dim)
    psi_hat = LieGroup.hat(psi)

    Psi = LieGroup.expm(psi_hat)

    assert torch.allclose(Psi, matrix_exp(psi_hat), atol=1e-6), "Exponential map failed to match torch.linalg.matrix_exp"

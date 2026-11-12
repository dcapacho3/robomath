"""Shared checks for the robomath tests."""

import numpy as np
from numpy.testing import assert_allclose


def assert_rotations(r: np.ndarray) -> None:
    """Every matrix of r is orthonormal: R R^T = I."""
    assert_allclose(
        r @ np.swapaxes(r, -1, -2),
        np.broadcast_to(np.eye(3), r.shape),
        rtol=0,
        atol=1e-12,
    )

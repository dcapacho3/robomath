"""Robomath - Test ZYX Euler angle conversions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from test_rotations import TOL

from robomath import euler_zyx_to_matrix, matrix_to_euler_zyx, rotx, roty, rotz

# A failure must be reproducible.
RNG = np.random.default_rng(0)
# Away from gimbal lock, inside the ranges the function returns.
INTERIOR = np.column_stack(
    [
        RNG.uniform(-np.pi + 0.1, np.pi - 0.1, 30),
        RNG.uniform(-1.4, 1.4, 30),
        RNG.uniform(-np.pi + 0.1, np.pi - 0.1, 30),
    ]
)
# Any orientation, including ones close to gimbal lock.
RANDOM_ROTATIONS = Rotation.random(100, rng=1).as_matrix()
# Loose enough for the lost precision just outside the lock branch.
NEAR_LOCK_TOL = 1e-7


@pytest.mark.parametrize("alpha, beta, gamma", INTERIOR)
def test_closed_form_matches_composition(alpha, beta, gamma):
    """euler_zyx_to_matrix equals rotz(a) @ roty(b) @ rotx(g)."""
    expected = rotz(alpha) @ roty(beta) @ rotx(gamma)
    assert_allclose(euler_zyx_to_matrix(alpha, beta, gamma), expected, atol=TOL)


@pytest.mark.parametrize("alpha, beta, gamma", INTERIOR)
def test_matches_scipy(alpha, beta, gamma):
    """Both directions match scipy's intrinsic ZYX convention."""
    rot = Rotation.from_euler("ZYX", [alpha, beta, gamma])
    assert_allclose(euler_zyx_to_matrix(alpha, beta, gamma), rot.as_matrix(), atol=TOL)
    assert_allclose(matrix_to_euler_zyx(rot.as_matrix()), rot.as_euler("ZYX"), atol=TOL)


@pytest.mark.parametrize(
    "angles",
    [
        (0.0, 0.0, 0.0),
        (np.pi / 2, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, -0.7),
        (0.3, -0.4, 1.2),
    ],
)
def test_known_angles_come_back(angles):
    """Angles away from gimbal lock are recovered unchanged."""
    out = matrix_to_euler_zyx(euler_zyx_to_matrix(*angles))
    assert_allclose(out, angles, atol=TOL)


@pytest.mark.parametrize("alpha, beta, gamma", INTERIOR)
def test_round_trip_angles(alpha, beta, gamma):
    """Angles then matrix then angles then matrix gives the same matrix."""
    m = euler_zyx_to_matrix(alpha, beta, gamma)
    assert_allclose(euler_zyx_to_matrix(*matrix_to_euler_zyx(m)), m, atol=TOL)


@pytest.mark.parametrize("m", RANDOM_ROTATIONS)
def test_round_trip_from_matrix(m):
    """Matrix then angles then matrix works for any orientation.

    The matrices are not built from angles, and some lie near gimbal lock.
    The angles also stay in the documented ranges.
    """
    alpha, beta, gamma = matrix_to_euler_zyx(m)
    assert_allclose(euler_zyx_to_matrix(alpha, beta, gamma), m, atol=NEAR_LOCK_TOL)
    assert -np.pi <= alpha <= np.pi
    assert -np.pi / 2 <= beta <= np.pi / 2
    assert -np.pi <= gamma <= np.pi


@pytest.mark.parametrize("beta", [np.pi / 2, -np.pi / 2])
@pytest.mark.parametrize(
    "alpha, gamma", [(0.3, 0.9), (-2.0, 1.5), (1.0, -1.0), (0.0, 0.0)]
)
def test_gimbal_lock(beta, alpha, gamma):
    """At beta = +-pi/2 the function sets alpha to 0 and still rebuilds R.

    Different angle triples give this matrix, so we compare matrixes
    """
    m = euler_zyx_to_matrix(alpha, beta, gamma)
    out_alpha, out_beta, out_gamma = matrix_to_euler_zyx(m)
    assert out_alpha == 0.0
    assert out_beta == pytest.approx(beta)
    assert np.isfinite(out_gamma)
    assert_allclose(euler_zyx_to_matrix(out_alpha, out_beta, out_gamma), m, atol=TOL)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("offset", [1e-9, 1e-7, 1e-5, 1e-3])
def test_just_inside_gimbal_lock(sign, offset):
    """Beta close to +-pi/2 still rebuilds R, on both sides of the lock tolerance."""
    m = euler_zyx_to_matrix(0.3, sign * (np.pi / 2 - offset), 0.9)
    assert_allclose(euler_zyx_to_matrix(*matrix_to_euler_zyx(m)), m, atol=NEAR_LOCK_TOL)

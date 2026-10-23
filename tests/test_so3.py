"""Robomath - Test SO(3) / so(3) maps and the plus and minus operators."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation
from test_rotations import TOL

from robomath import (
    exp_so3,
    exp_vec,
    hat,
    lminus,
    log_so3,
    log_vec,
    lplus,
    rminus,
    rplus,
    vee,
)

# Fill in the tests, then remove the skip marker of each one.
# Pattern from test_quaternions.py: scipy as the outside check, matrices
# compared (not vectors) near pi where the axis sign is free.
RNG = np.random.default_rng(0)
VECTORS = RNG.normal(size=(30, 3))  # lengths mostly in [0, pi]; some larger
PARTNERS = np.roll(VECTORS, 1, axis=0)
ROTATIONS = Rotation.random(30, rng=1)
# Special lengths: zero, tiny, and exactly pi.
SPECIAL = [np.zeros(3), np.array([1e-9, 0, 0]), np.array([0, 0, np.pi])]
# VECTORS directions, lengths 0.1 .. 3.0: all below pi, where Log is unique.
INSIDE = VECTORS / np.linalg.norm(VECTORS, axis=1, keepdims=True)
INSIDE = INSIDE * np.linspace(0.1, 3.0, 30)[:, None]
MATRICES = ROTATIONS.as_matrix()
OTHERS = np.roll(MATRICES, 1, axis=0)  # a different rotation for each matrix
SMALL = VECTORS * 0.3  # lengths below pi, so Log does not wrap


@pytest.mark.parametrize("v, u", list(zip(VECTORS, PARTNERS, strict=True)))
def test_hat_is_cross_product(v, u):
    """hat(v) @ u == np.cross(v, u), and hat(v).T == -hat(v)."""
    assert_allclose(hat(v) @ u, np.cross(v, u), atol=TOL)
    assert_allclose(hat(v).T, -hat(v), atol=TOL)


@pytest.mark.parametrize("v", VECTORS)
def test_vee_inverts_hat(v):
    """vee(hat(v)) == v for the VECTORS."""
    assert_allclose(vee(hat(v)), v, atol=TOL)


@pytest.mark.parametrize("v", [*VECTORS, *SPECIAL])
def test_exp_vec_matches_scipy(v):
    """exp_vec(v) == Rotation.from_rotvec(v).as_matrix(). Includes v = 0."""
    assert_allclose(exp_vec(v), Rotation.from_rotvec(v).as_matrix(), atol=TOL)


@pytest.mark.parametrize(
    "r, rotvec", list(zip(ROTATIONS.as_matrix(), ROTATIONS.as_rotvec(), strict=True))
)
def test_log_vec_matches_scipy(r, rotvec):
    """log_vec(R) == Rotation.as_rotvec() for the ROTATIONS."""
    assert_allclose(log_vec(r), rotvec, atol=TOL)


def test_log_of_identity_is_zero():
    """R = I has no axis, but its rotation vector is 0 (axis * angle, angle = 0)."""
    assert_allclose(log_vec(np.eye(3)), np.zeros(3), atol=TOL)


@pytest.mark.parametrize("v", INSIDE)
def test_log_of_exp_round_trip(v):
    """log_vec(exp_vec(v)) == v for |v| < pi."""
    assert_allclose(log_vec(exp_vec(v)), v, atol=TOL)


def test_log_of_exp_tiny_vector():
    """A 1e-9 vector survives the round trip (relative error, not absolute)."""
    v = np.array([1e-9, 0, 0])
    assert_allclose(log_vec(exp_vec(v)), v, rtol=1e-6, atol=0)


@pytest.mark.parametrize("v", [*VECTORS, *SPECIAL])
def test_exp_so3_is_matrix_exponential(v):
    """exp_so3(hat(v)) equals scipy's general matrix exponential and exp_vec(v)."""
    assert_allclose(exp_so3(hat(v)), expm(hat(v)), atol=TOL)
    assert_allclose(exp_so3(hat(v)), exp_vec(v), atol=TOL)


@pytest.mark.parametrize("v", INSIDE)
def test_log_so3_is_matrix_logarithm(v):
    """log_so3 gives back hat(v), and matches scipy's general matrix logarithm."""
    r = exp_vec(v)
    assert_allclose(log_so3(r), hat(v), atol=TOL)
    assert_allclose(log_so3(r), logm(r).real, atol=1e-9)  # logm is less exact


@pytest.mark.parametrize("r, tau", list(zip(MATRICES, SMALL, strict=True)))
def test_rplus_rminus_round_trip(r, tau):
    """rminus(rplus(R, tau), R) == tau, for small tau (eq. 25, 26)."""
    assert_allclose(rminus(rplus(r, tau), r), tau, atol=TOL)


@pytest.mark.parametrize("r1, r2", list(zip(MATRICES, OTHERS, strict=True)))
def test_rminus_takes_first_argument_to_second(r1, r2):
    """rplus(R1, rminus(R2, R1)) == R2: the definition of minus."""
    assert_allclose(rplus(r1, rminus(r2, r1)), r2, atol=TOL)


@pytest.mark.parametrize("r, tau", list(zip(MATRICES, SMALL, strict=True)))
def test_left_and_right_agree_through_adjoint(r, tau):
    """lplus(R @ tau, R) == rplus(R, tau): the adjoint of SO(3) is R (eq. 32, 139)."""
    assert_allclose(lplus(r @ tau, r), rplus(r, tau), atol=TOL)


@pytest.mark.parametrize("r, tau", list(zip(MATRICES, SMALL, strict=True)))
def test_lminus_round_trip(r, tau):
    """lminus(lplus(tau, R), R) == tau."""
    assert_allclose(lminus(lplus(tau, r), r), tau, atol=TOL)


@pytest.mark.parametrize("r1, r2", list(zip(MATRICES, OTHERS, strict=True)))
def test_lminus_takes_first_argument_to_second(r1, r2):
    """lplus(lminus(R2, R1), R1) == R2: the definition of left minus."""
    assert_allclose(lplus(lminus(r2, r1), r1), r2, atol=TOL)


def test_plus_minus_differ_between_left_and_right():
    """With a non-trivial R, the same tau gives different results on each side."""
    r, tau = MATRICES[0], SMALL[0]
    assert not np.allclose(lplus(tau, r), rplus(r, tau))

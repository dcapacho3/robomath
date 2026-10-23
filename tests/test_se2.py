"""Robomath - Test SO(2) and SE(2) against SE(3) and the group laws."""

import numpy as np
from numpy.testing import assert_allclose
from test_rotations import TOL

from robomath import (
    adjoint_se2,
    exp_so2,
    exp_vec_se2,
    exp_vec_se3,
    hat_se2,
    lminus_se2,
    log_so2,
    log_vec_se2,
    lplus_se2,
    rminus_se2,
    rp_to_trans,
    rplus_se2,
    trans_inv,
    vee_se2,
    wrap_angle,
)

RNG = np.random.default_rng(0)
ANGLES = RNG.uniform(-np.pi, np.pi, 30)
POSES = rp_to_trans(exp_so2(ANGLES), RNG.normal(size=(30, 2)))
OTHERS = np.roll(POSES, 1, axis=0)
TAUS = RNG.normal(size=(30, 3))
TAUS[:, 2] = np.linspace(-3.0, 3.0, 30)  # includes angles near 0 and near pi


def embed(tau):
    """The same motion as a tangent vector of SE(3): (x, y, 0, 0, 0, theta)."""
    return np.array([tau[0], tau[1], 0, 0, 0, tau[2]])


def test_so2_round_trip_and_matrix():
    """log_so2(exp_so2(a)) == a, and exp_so2 is [[c, -s], [s, c]]."""
    assert_allclose(log_so2(exp_so2(ANGLES)), ANGLES, rtol=0, atol=TOL)
    assert_allclose(exp_so2(np.pi / 2), [[0, -1], [1, 0]], rtol=0, atol=TOL)


def test_wrap_angle():
    """Angles outside (-pi, pi] come back to the same direction inside it."""
    assert_allclose(wrap_angle(ANGLES + 4 * np.pi), ANGLES, rtol=0, atol=1e-12)
    assert_allclose(wrap_angle(3 * np.pi / 2), -np.pi / 2, rtol=0, atol=TOL)


def test_hat_and_vee_se2():
    """vee_se2(hat_se2(tau)) == tau, and the matrix has the paper's layout."""
    assert_allclose(vee_se2(hat_se2(TAUS)), TAUS, rtol=0, atol=0)
    assert_allclose(
        hat_se2(np.array([1.0, 2.0, 3.0])),
        [[0, -3, 1], [3, 0, 2], [0, 0, 0]],
        rtol=0,
        atol=0,
    )


def test_exp_vec_se2_matches_se3():
    """SE(2) is SE(3) with motion in the xy plane: compare with exp_vec_se3."""
    for tau in TAUS:
        full = exp_vec_se3(embed(tau))
        expected = full[np.ix_([0, 1, 3], [0, 1, 3])]
        assert_allclose(exp_vec_se2(tau), expected, rtol=0, atol=TOL)


def test_exp_vec_se2_small_angle():
    """At theta = 0 and theta = 1e-9 the pose is a pure translation by rho."""
    for theta in (0.0, 1e-9):
        m = exp_vec_se2(np.array([1.0, -2.0, theta]))
        assert_allclose(m[:2, 2], [1.0, -2.0], rtol=0, atol=1e-8)
        assert_allclose(m[:2, :2], np.eye(2), rtol=0, atol=1e-8)


def test_log_of_exp_round_trip():
    """log_vec_se2(exp_vec_se2(tau)) == tau for |theta| < pi, also at theta = 0."""
    assert_allclose(log_vec_se2(exp_vec_se2(TAUS)), TAUS, rtol=0, atol=1e-12)
    zero = np.array([1.0, 2.0, 0.0])
    assert_allclose(log_vec_se2(exp_vec_se2(zero)), zero, rtol=0, atol=TOL)


def test_log_near_pi():
    """The inverse of V stays accurate at the edge of the range."""
    tau = np.array([1.0, -2.0, np.pi - 1e-9])
    assert_allclose(log_vec_se2(exp_vec_se2(tau)), tau, rtol=0, atol=1e-8)


def test_trans_inv_works_on_se2_poses():
    """M @ trans_inv(M) is the 3x3 identity."""
    assert_allclose(
        POSES @ trans_inv(POSES),
        np.broadcast_to(np.eye(3), (30, 3, 3)),
        rtol=0,
        atol=TOL,
    )


def test_adjoint_definition_and_product():
    """exp(Ad(M) tau) == M exp(tau) M^-1, and Ad is a homomorphism."""
    lhs = exp_vec_se2(np.einsum("nij,nj->ni", adjoint_se2(POSES), TAUS))
    rhs = POSES @ exp_vec_se2(TAUS) @ trans_inv(POSES)
    assert_allclose(lhs, rhs, rtol=0, atol=1e-12)
    assert_allclose(
        adjoint_se2(POSES @ OTHERS),
        adjoint_se2(POSES) @ adjoint_se2(OTHERS),
        rtol=0,
        atol=TOL,
    )


def test_plus_minus():
    """Round trips, the definition of minus, and left/right agreement."""
    assert_allclose(rminus_se2(rplus_se2(POSES, TAUS), POSES), TAUS, rtol=0, atol=1e-12)
    assert_allclose(lminus_se2(lplus_se2(TAUS, POSES), POSES), TAUS, rtol=0, atol=1e-12)
    assert_allclose(
        rplus_se2(POSES, rminus_se2(OTHERS, POSES)), OTHERS, rtol=0, atol=1e-12
    )
    assert_allclose(
        lplus_se2(lminus_se2(OTHERS, POSES), POSES), OTHERS, rtol=0, atol=1e-12
    )
    local = np.einsum("nij,nj->ni", adjoint_se2(POSES), TAUS)
    assert_allclose(lplus_se2(local, POSES), rplus_se2(POSES, TAUS), rtol=0, atol=1e-12)

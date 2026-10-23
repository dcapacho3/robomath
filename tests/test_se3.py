"""Robomath - Test the SE(3) functions against modern_robotics and the paper."""

import modern_robotics as mr
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from test_rotations import TOL

from robomath import (
    adjoint,
    angular,
    exp_vec_se3,
    hat_se3,
    linear,
    lminus_se3,
    log_vec_se3,
    lplus_se3,
    rminus_se3,
    rp_to_trans,
    rplus_se3,
    trans_inv,
    trans_to_rp,
    twist,
    vee_se3,
)

RNG = np.random.default_rng(0)
ROTATIONS = Rotation.random(30, rng=1).as_matrix()
TRANSLATIONS = RNG.normal(size=(30, 3))
POSES = rp_to_trans(ROTATIONS, TRANSLATIONS)
# Tangent vectors (rho, theta): rotation parts of length 0.1 .. 3.0 (below pi).
TAUS = RNG.normal(size=(30, 6))
TAUS[:, 3:] *= (np.linspace(0.1, 3.0, 30) / np.linalg.norm(TAUS[:, 3:], axis=1))[
    :, None
]
OTHERS = np.roll(POSES, 1, axis=0)
# The paper orders a tangent vector (rho, theta); modern_robotics orders it
# (theta, rho). SWAP exchanges the halves.
SWAP = np.block([[np.zeros((3, 3)), np.eye(3)], [np.eye(3), np.zeros((3, 3))]])


def to_mr(tau):
    """Reorder (rho, theta) into modern_robotics' (theta, rho)."""
    return SWAP @ tau


def test_rp_to_trans_and_back():
    """trans_to_rp(rp_to_trans(R, t)) gives R and t; matches mr.RpToTrans."""
    for r, t, m in zip(ROTATIONS, TRANSLATIONS, POSES, strict=True):
        assert_allclose(m, mr.RpToTrans(r, t), rtol=0, atol=TOL)
        r2, t2 = trans_to_rp(m)
        assert_allclose(r2, r, rtol=0, atol=TOL)
        assert_allclose(t2, t, rtol=0, atol=TOL)


def test_trans_to_rp_returns_copies():
    """Editing the returned blocks leaves the pose unchanged."""
    m = POSES[0].copy()
    r, t = trans_to_rp(m)
    r[:] = 0
    t[:] = 0
    assert_allclose(m, POSES[0], rtol=0, atol=0)


def test_trans_inv():
    """trans_inv matches mr.TransInv, and M @ trans_inv(M) is the identity."""
    for m in POSES:
        assert_allclose(trans_inv(m), mr.TransInv(m), rtol=0, atol=TOL)
        assert_allclose(m @ trans_inv(m), np.eye(4), rtol=0, atol=TOL)


def test_hat_and_vee_se3():
    """vee_se3(hat_se3(tau)) == tau; hat_se3 matches mr.VecTose3(to_mr(tau))."""
    for tau in TAUS:
        assert_allclose(vee_se3(hat_se3(tau)), tau, rtol=0, atol=TOL)
        assert_allclose(hat_se3(tau), mr.VecTose3(to_mr(tau)), rtol=0, atol=TOL)


def test_exp_vec_se3_matches_modern_robotics():
    """exp_vec_se3(tau) == mr.MatrixExp6(mr.VecTose3(to_mr(tau)))."""
    for tau in TAUS:
        expected = mr.MatrixExp6(mr.VecTose3(to_mr(tau)))
        assert_allclose(exp_vec_se3(tau), expected, rtol=0, atol=TOL)


@pytest.mark.parametrize("size", [0.0, 1e-9])
def test_exp_vec_se3_pure_translation(size):
    """With theta = 0 (or tiny) the pose is a translation by rho, no rotation."""
    rho = np.array([1.0, -2.0, 3.0])
    m = exp_vec_se3(twist(rho, np.array([size, 0, 0])))
    assert_allclose(m[:3, :3], np.eye(3), rtol=0, atol=1e-8)
    assert_allclose(m[:3, 3], rho, rtol=0, atol=1e-8)


def test_log_vec_se3_matches_modern_robotics():
    """log_vec_se3(M) == to_mr(mr.se3ToVec(mr.MatrixLog6(M)))."""
    for m in POSES:
        expected = to_mr(mr.se3ToVec(mr.MatrixLog6(m)))
        assert_allclose(log_vec_se3(m), expected, rtol=0, atol=1e-9)


def test_log_of_exp_round_trip():
    """log_vec_se3(exp_vec_se3(tau)) == tau for the TAUS (rotation part below pi)."""
    assert_allclose(log_vec_se3(exp_vec_se3(TAUS)), TAUS, rtol=0, atol=1e-9)


def test_adjoint_matches_modern_robotics():
    """adjoint(M) == SWAP @ mr.Adjoint(M) @ SWAP: same map, other vector order."""
    for m in POSES:
        assert_allclose(adjoint(m), SWAP @ mr.Adjoint(m) @ SWAP, rtol=0, atol=TOL)


def test_adjoint_definition():
    """exp_vec_se3(adjoint(M) @ tau) == M @ exp_vec_se3(tau) @ trans_inv(M) (eq. 29)."""
    for m, tau in zip(POSES, TAUS, strict=True):
        lhs = exp_vec_se3(adjoint(m) @ tau)
        assert_allclose(lhs, m @ exp_vec_se3(tau) @ trans_inv(m), rtol=0, atol=1e-9)


def test_adjoint_of_a_product():
    """adjoint(M1 @ M2) == adjoint(M1) @ adjoint(M2) (a homomorphism)."""
    assert_allclose(
        adjoint(POSES @ OTHERS), adjoint(POSES) @ adjoint(OTHERS), rtol=0, atol=TOL
    )


def test_plus_minus_round_trips():
    """rminus_se3(rplus_se3(M, tau), M) == tau; same for lminus_se3 and lplus_se3."""
    assert_allclose(rminus_se3(rplus_se3(POSES, TAUS), POSES), TAUS, rtol=0, atol=1e-9)
    assert_allclose(lminus_se3(lplus_se3(TAUS, POSES), POSES), TAUS, rtol=0, atol=1e-9)


def test_minus_takes_first_argument_to_second():
    """rplus(M1, rminus(M2, M1)) == M2, and the same on the left."""
    assert_allclose(
        rplus_se3(POSES, rminus_se3(OTHERS, POSES)), OTHERS, rtol=0, atol=1e-9
    )
    assert_allclose(
        lplus_se3(lminus_se3(OTHERS, POSES), POSES), OTHERS, rtol=0, atol=1e-9
    )


def test_left_and_right_agree_through_the_adjoint():
    """lplus_se3(adjoint(M) @ tau, M) == rplus_se3(M, tau) (eq. 32)."""
    for m, tau in zip(POSES, TAUS, strict=True):
        assert_allclose(
            lplus_se3(adjoint(m) @ tau, m), rplus_se3(m, tau), rtol=0, atol=1e-9
        )


def test_twist_linear_angular_round_trip():
    """linear(twist(a, b)) is a and angular(twist(a, b)) is b, also for batches."""
    a, b = TRANSLATIONS, TAUS[:, 3:]
    assert_allclose(linear(twist(a, b)), a, rtol=0, atol=0)
    assert_allclose(angular(twist(a, b)), b, rtol=0, atol=0)
    assert_allclose(linear(twist(a[0], b[0])), a[0], rtol=0, atol=0)


def test_layout_sentinels():
    """The tangent layout is (translation, rotation), checked by two extremes."""
    move = exp_vec_se3(np.array([1.0, 0, 0, 0, 0, 0]))
    assert_allclose(move, rp_to_trans(np.eye(3), [1.0, 0, 0]), rtol=0, atol=TOL)
    turn = exp_vec_se3(np.array([0, 0, 0, 1.0, 0, 0]))
    expected = rp_to_trans(Rotation.from_rotvec([1.0, 0, 0]).as_matrix(), np.zeros(3))
    assert_allclose(turn, expected, rtol=0, atol=TOL)

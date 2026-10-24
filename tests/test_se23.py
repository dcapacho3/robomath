"""Robomath - Test SE_2(3) against scipy's matrix exponential and the group laws."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm, logm
from test_rotations import TOL

import robomath as rm

RNG = np.random.default_rng(0)
TAUS = RNG.normal(size=(30, 9))
TAUS[:, 3:6] *= (np.linspace(0.1, 3.0, 30) / np.linalg.norm(TAUS[:, 3:6], axis=1))[
    :, None
]
POSES = rm.exp_vec_se23(TAUS)
OTHERS = np.roll(POSES, 1, axis=0)
EYE = np.broadcast_to(np.eye(5), POSES.shape)


def test_hat_and_vee_se23():
    """vee(hat(tau)) == tau, and the columns are [theta]x, nu, rho."""
    assert_allclose(rm.vee_se23(rm.hat_se23(TAUS)), TAUS, rtol=0, atol=TOL)
    m = rm.hat_se23(np.array([1.0, 2.0, 3.0, 0, 0, 0.5, 4.0, 5.0, 6.0]))
    assert_allclose(m[:3, 4], [1, 2, 3], rtol=0, atol=0)
    assert_allclose(m[:3, 3], [4, 5, 6], rtol=0, atol=0)
    assert_allclose(m[:3, :3], rm.hat(np.array([0, 0, 0.5])), rtol=0, atol=0)


def test_exp_matches_matrix_exponential():
    """exp_vec_se23 equals scipy's expm of the hat matrix."""
    for tau, m in zip(TAUS, POSES, strict=True):
        assert_allclose(m, expm(rm.hat_se23(tau)), rtol=0, atol=1e-12)


def test_exp_without_velocity_is_se3():
    """With nu = 0 the rotation and position are those of SE(3), the velocity zero."""
    tau = TAUS.copy()
    tau[:, 6:9] = 0.0
    r, v, p = rm.se23_to_rvp(rm.exp_vec_se23(tau))
    pose = rm.exp_vec_se3(tau[:, :6])
    assert_allclose(r, pose[:, :3, :3], rtol=0, atol=1e-13)
    assert_allclose(p, pose[:, :3, 3], rtol=0, atol=1e-13)
    assert_allclose(v, 0.0, rtol=0, atol=0)


def test_log_round_trip_and_matches_scipy():
    """log(exp(tau)) == tau; the same vector comes out of scipy's logm."""
    assert_allclose(rm.log_vec_se23(POSES), TAUS, rtol=0, atol=1e-11)
    for m, tau in zip(POSES[:10], TAUS[:10], strict=True):
        assert_allclose(rm.vee_se23(logm(m).real), tau, rtol=0, atol=1e-8)


def test_rvp_round_trip_and_inverse():
    """The parts come back, the copies are independent, and M @ M^-1 = I."""
    r, v, p = rm.se23_to_rvp(POSES)
    assert_allclose(rm.rvp_to_se23(r, v, p), POSES, rtol=0, atol=0)
    r[:] = 0
    assert_allclose(rm.se23_to_rvp(POSES)[0][0], POSES[0, :3, :3], rtol=0, atol=0)
    assert_allclose(POSES @ rm.se23_inv(POSES), EYE, rtol=0, atol=1e-12)


def test_adjoint_definition_and_product():
    """exp(Ad(M) tau) == M exp(tau) M^-1, and Ad is a homomorphism."""
    for m, tau in zip(POSES, TAUS * 0.3, strict=True):
        lhs = rm.exp_vec_se23(rm.adjoint_se23(m) @ tau)
        assert_allclose(
            lhs, m @ rm.exp_vec_se23(tau) @ rm.se23_inv(m), rtol=0, atol=1e-11
        )
    assert_allclose(
        rm.adjoint_se23(POSES @ OTHERS),
        rm.adjoint_se23(POSES) @ rm.adjoint_se23(OTHERS),
        rtol=0,
        atol=1e-11,
    )


def test_plus_minus():
    """Round trips, the definition of minus, and left/right agreement."""
    small = TAUS * 0.3
    assert_allclose(
        rm.rminus_se23(rm.rplus_se23(POSES, small), POSES), small, rtol=0, atol=1e-11
    )
    assert_allclose(
        rm.lminus_se23(rm.lplus_se23(small, POSES), POSES), small, rtol=0, atol=1e-11
    )
    assert_allclose(
        rm.rplus_se23(POSES, rm.rminus_se23(OTHERS, POSES)), OTHERS, rtol=0, atol=1e-11
    )
    assert_allclose(
        rm.lplus_se23(rm.lminus_se23(OTHERS, POSES), POSES), OTHERS, rtol=0, atol=1e-11
    )
    local = np.einsum("nij,nj->ni", rm.adjoint_se23(POSES), small)
    assert_allclose(
        rm.lplus_se23(local, POSES), rm.rplus_se23(POSES, small), rtol=0, atol=1e-11
    )

"""Robomath - Test Sim(3) against scipy's matrix exponential and the group laws."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation
from test_rotations import TOL

import robomath as rm

RNG = np.random.default_rng(0)
TAUS = RNG.normal(size=(30, 7))
# Rotation parts of length 0.1 .. 3.0 (below pi); scales between e^-1 and e^1.
TAUS[:, 3:6] *= (np.linspace(0.1, 3.0, 30) / np.linalg.norm(TAUS[:, 3:6], axis=1))[
    :, None
]
TAUS[:, 6] = np.clip(TAUS[:, 6], -1, 1)
POSES = rm.exp_vec_sim3(TAUS)
OTHERS = np.roll(POSES, 1, axis=0)
SMALL = TAUS * 0.3


def test_hat_and_vee_sim3():
    """vee(hat(tau)) == tau, and hat is [[sigma I + [theta]x, rho], [0, 0]]."""
    assert_allclose(rm.vee_sim3(rm.hat_sim3(TAUS)), TAUS, rtol=0, atol=TOL)
    m = rm.hat_sim3(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.5, 0.25]))
    expected = [[0.25, -0.5, 0, 1], [0.5, 0.25, 0, 2], [0, 0, 0.25, 3], [0, 0, 0, 0]]
    assert_allclose(m, expected, rtol=0, atol=0)


def test_exp_matches_matrix_exponential():
    """exp_vec_sim3 equals scipy's expm of the hat matrix."""
    for tau, m in zip(TAUS, POSES, strict=True):
        assert_allclose(m, expm(rm.hat_sim3(tau)), rtol=1e-11, atol=1e-11)


def test_exp_with_no_scale_is_se3():
    """With sigma = 0 the pose is the SE(3) pose."""
    tau = TAUS.copy()
    tau[:, 6] = 0.0
    assert_allclose(
        rm.exp_vec_sim3(tau), rm.exp_vec_se3(tau[:, :6]), rtol=0, atol=1e-12
    )


def test_exp_pure_scale_and_small_vectors():
    """With theta = 0 the translation is rho (e^sigma - 1) / sigma; tiny values hold."""
    sigma = 0.7
    tau = np.array([1.0, -2.0, 3.0, 0, 0, 0, sigma])
    scale = np.expm1(sigma) / sigma
    assert_allclose(rm.exp_vec_sim3(tau)[:3, 3], tau[:3] * scale, rtol=0, atol=1e-13)
    tiny = np.array([1.0, -2.0, 3.0, 1e-9, 0, 0, 1e-9])
    assert_allclose(rm.exp_vec_sim3(tiny)[:3, 3], tiny[:3], rtol=0, atol=1e-8)
    assert_allclose(rm.exp_vec_sim3(np.zeros(7)), np.eye(4), rtol=0, atol=0)


def test_log_round_trip_and_matches_scipy():
    """log(exp(tau)) == tau; the same vector comes out of scipy's logm."""
    assert_allclose(rm.log_vec_sim3(POSES), TAUS, rtol=0, atol=1e-11)
    for m, tau in zip(POSES[:10], TAUS[:10], strict=True):
        assert_allclose(rm.vee_sim3(logm(m).real), tau, rtol=0, atol=1e-8)


def test_srt_round_trip_and_composition():
    """Scale, rotation and translation come back; scales multiply under @."""
    s, r, t = rm.sim3_to_srt(POSES)
    assert_allclose(rm.srt_to_sim3(s, r, t), POSES, rtol=0, atol=1e-12)
    assert_allclose(
        r @ np.swapaxes(r, -1, -2),
        np.broadcast_to(np.eye(3), r.shape),
        rtol=0,
        atol=1e-12,
    )
    s2, _, _ = rm.sim3_to_srt(POSES @ OTHERS)
    assert_allclose(s2, s * rm.sim3_to_srt(OTHERS)[0], rtol=1e-12, atol=0)


def test_inverse():
    """M @ sim3_inv(M) is the identity, and the scale inverts."""
    assert_allclose(
        POSES @ rm.sim3_inv(POSES),
        np.broadcast_to(np.eye(4), POSES.shape),
        rtol=0,
        atol=1e-12,
    )
    assert_allclose(
        rm.sim3_to_srt(rm.sim3_inv(POSES))[0],
        1 / rm.sim3_to_srt(POSES)[0],
        rtol=1e-12,
        atol=0,
    )


def test_adjoint_definition_and_product():
    """exp(Ad(M) tau) == M exp(tau) M^-1, and Ad is a homomorphism."""
    for m, tau in zip(POSES, SMALL, strict=True):
        lhs = rm.exp_vec_sim3(rm.adjoint_sim3(m) @ tau)
        assert_allclose(
            lhs, m @ rm.exp_vec_sim3(tau) @ rm.sim3_inv(m), rtol=1e-9, atol=1e-9
        )
    assert_allclose(
        rm.adjoint_sim3(POSES @ OTHERS),
        rm.adjoint_sim3(POSES) @ rm.adjoint_sim3(OTHERS),
        rtol=1e-11,
        atol=1e-11,
    )


def test_plus_minus():
    """Round trips, the definition of minus, and left/right agreement."""
    assert_allclose(
        rm.rminus_sim3(rm.rplus_sim3(POSES, SMALL), POSES), SMALL, rtol=0, atol=1e-11
    )
    assert_allclose(
        rm.lminus_sim3(rm.lplus_sim3(SMALL, POSES), POSES), SMALL, rtol=0, atol=1e-11
    )
    assert_allclose(
        rm.rplus_sim3(POSES, rm.rminus_sim3(OTHERS, POSES)),
        OTHERS,
        rtol=1e-11,
        atol=1e-11,
    )
    assert_allclose(
        rm.lplus_sim3(rm.lminus_sim3(OTHERS, POSES), POSES),
        OTHERS,
        rtol=1e-11,
        atol=1e-11,
    )
    local = np.einsum("nij,nj->ni", rm.adjoint_sim3(POSES), SMALL)
    assert_allclose(
        rm.lplus_sim3(local, POSES), rm.rplus_sim3(POSES, SMALL), rtol=1e-9, atol=1e-9
    )


def test_umeyama_gives_a_sim3_that_maps_the_points():
    """The fit from umeyama, as a Sim(3) pose, carries p onto q."""
    p = RNG.normal(size=(12, 3))
    r = Rotation.random(rng=3).as_matrix()
    q = 1.7 * p @ r.T + np.array([0.5, -1.0, 2.0])
    m = rm.srt_to_sim3(*rm.umeyama(p, q))
    moved = (m[:3, :3] @ p.T).T + m[:3, 3]
    assert_allclose(moved, q, rtol=0, atol=1e-12)

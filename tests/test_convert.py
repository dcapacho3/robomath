"""Robomath - Test the layout conversions against scipy and modern_robotics."""

import modern_robotics as mr
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from test_rotations import TOL

import robomath as rm

RNG = np.random.default_rng(0)
ROTATIONS = Rotation.random(30, rng=1)
TRANSLATIONS = RNG.normal(size=(30, 3))
TAUS = RNG.normal(size=(30, 6))


def test_quaternion_order_changes_only_the_scalar_position():
    """(x, y, z, w) -> (w, x, y, z), checked on known numbers and as a round trip."""
    q = np.array([0.1, 0.2, 0.3, 0.9])
    assert_allclose(rm.quat_xyzw_to_wxyz(q), [0.9, 0.1, 0.2, 0.3], rtol=0, atol=0)
    assert_allclose(rm.quat_wxyz_to_xyzw(q), [0.2, 0.3, 0.9, 0.1], rtol=0, atol=0)
    quats = ROTATIONS.as_quat()
    assert_allclose(
        rm.quat_wxyz_to_xyzw(rm.quat_xyzw_to_wxyz(quats)), quats, rtol=0, atol=0
    )


def test_quaternion_order_matches_scipy():
    """Scipy's scalar_first option gives (w, x, y, z), the order of Drake and MuJoCo."""
    assert_allclose(
        rm.quat_xyzw_to_wxyz(ROTATIONS.as_quat()),
        ROTATIONS.as_quat(scalar_first=True),
        rtol=0,
        atol=0,
    )


def test_swap_twist_exchanges_the_halves():
    """(1, 2, 3, 4, 5, 6) -> (4, 5, 6, 1, 2, 3), and swapping twice is the identity."""
    tau = np.arange(1.0, 7.0)
    assert_allclose(rm.swap_twist(tau), [4, 5, 6, 1, 2, 3], rtol=0, atol=0)
    assert_allclose(rm.swap_twist(rm.swap_twist(TAUS)), TAUS, rtol=0, atol=0)


def test_swap_twist_gives_the_same_pose_as_modern_robotics():
    """Exp of the swapped vector in modern_robotics equals exp_vec_se3 here."""
    for tau in TAUS:
        pose = mr.MatrixExp6(mr.VecTose3(rm.swap_twist(tau)))
        assert_allclose(pose, rm.exp_vec_se3(tau), rtol=0, atol=1e-12)


def test_pose_and_position_quaternion_round_trip():
    """pq_to_trans builds the pose scipy would; trans_to_pq gives the same rotation."""
    poses = rm.pq_to_trans(TRANSLATIONS, ROTATIONS.as_quat())
    assert_allclose(poses[:, :3, :3], ROTATIONS.as_matrix(), rtol=0, atol=TOL)
    assert_allclose(poses[:, :3, 3], TRANSLATIONS, rtol=0, atol=0)
    p, q = rm.trans_to_pq(poses)
    assert_allclose(p, TRANSLATIONS, rtol=0, atol=0)
    assert_allclose(rm.quat_to_matrix(q), ROTATIONS.as_matrix(), rtol=0, atol=TOL)

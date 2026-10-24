"""Robomath - Test the Transform class against the SE(3) functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from test_rotations import TOL

import robomath as rm
from robomath import Transform

RNG = np.random.default_rng(0)
TAUS = RNG.normal(size=(30, 6))
TAUS[:, 3:] *= (np.linspace(0.1, 3.0, 30) / np.linalg.norm(TAUS[:, 3:], axis=1))[
    :, None
]
POSES = rm.exp_vec_se3(TAUS)
T = Transform(POSES)
OTHER = Transform(np.roll(POSES, 1, axis=0))


def test_constructor_checks_shape_and_stores_a_read_only_copy():
    """Wrong shapes raise; the stored matrix is a float copy that cannot be edited."""
    with pytest.raises(ValueError, match="4, 4"):
        Transform(np.eye(3))
    source = np.eye(4, dtype=int)
    t = Transform(source)
    source[0, 3] = 9
    assert t.matrix[0, 3] == 0.0
    assert t.matrix.dtype == float
    with pytest.raises(ValueError, match="read-only"):
        t.matrix[0, 0] = 2.0
    assert Transform.identity().matrix.shape == (4, 4)
    assert T.shape == (30,)


def test_constructors_agree():
    """from_rt, from_pq and exp build the pose the functions build."""
    r, t = rm.trans_to_rp(POSES)
    assert_allclose(Transform.from_rt(r, t).matrix, POSES, rtol=0, atol=0)
    assert_allclose(Transform.exp(TAUS).matrix, POSES, rtol=0, atol=0)
    p, q = rm.trans_to_pq(POSES)
    assert_allclose(Transform.from_pq(p, q).matrix, POSES, rtol=0, atol=TOL)
    assert_allclose(T.rotation, r, rtol=0, atol=0)
    assert_allclose(T.translation, t, rtol=0, atol=0)


def test_from_matrix_repairs_a_drifted_pose():
    """Noise in the rotation and a wrong bottom row come out as a true pose."""
    noisy = POSES + 1e-3 * RNG.normal(size=POSES.shape)
    fixed = Transform.from_matrix(noisy)
    r = fixed.rotation
    assert_allclose(
        r @ np.swapaxes(r, -1, -2),
        np.broadcast_to(np.eye(3), r.shape),
        rtol=0,
        atol=1e-12,
    )
    assert_allclose(np.linalg.det(r), 1.0, rtol=0, atol=1e-12)
    assert_allclose(fixed.matrix[:, 3], [[0, 0, 0, 1]] * 30, rtol=0, atol=0)
    expected = Rotation.from_matrix(noisy[:, :3, :3]).as_matrix()
    assert_allclose(r, expected, rtol=0, atol=1e-9)


def test_matmul_composes_and_inverse_undoes():
    """T_ab @ T_bc is the matrix product, and T @ T.inverse() is the identity."""
    assert_allclose((T @ OTHER).matrix, T.matrix @ OTHER.matrix, rtol=0, atol=0)
    assert isinstance(T @ OTHER, Transform)
    identity = Transform(np.broadcast_to(np.eye(4), T.matrix.shape))
    assert (T @ T.inverse()).isclose(identity, atol=1e-12)
    assert (T.inverse() @ T).isclose(identity, atol=1e-12)


def test_matmul_moves_points():
    """T @ p is R p + t for one point, a list of points, and a batch of poses."""
    one = Transform(POSES[0])
    p = RNG.normal(size=3)
    assert_allclose(one @ p, POSES[0, :3, :3] @ p + POSES[0, :3, 3], rtol=0, atol=TOL)
    pts = RNG.normal(size=(8, 3))
    expected = pts @ POSES[0, :3, :3].T + POSES[0, :3, 3]
    assert_allclose(one @ pts, expected, rtol=0, atol=TOL)
    one_each = RNG.normal(size=(30, 3))
    moved = T @ one_each
    assert_allclose(
        moved[7], POSES[7, :3, :3] @ one_each[7] + POSES[7, :3, 3], rtol=0, atol=TOL
    )
    # Moving by T and then by T.inverse() gives the points back.
    assert_allclose(T.inverse() @ moved, one_each, rtol=0, atol=1e-12)


def test_matmul_rejects_what_it_cannot_use():
    """A wrong shape, and an array on the left, raise TypeError, not a silent result."""
    with pytest.raises(TypeError):
        T @ np.ones(4)
    with pytest.raises(TypeError):
        np.ones(3) @ T


def test_log_adjoint_and_plus_minus():
    """The methods give what the functions give."""
    assert_allclose(T.log(), TAUS, rtol=0, atol=1e-11)
    assert_allclose(T.adjoint(), rm.adjoint(POSES), rtol=0, atol=0)
    small = TAUS * 0.3
    assert_allclose(T.rplus(small).rminus(T), small, rtol=0, atol=1e-11)
    assert_allclose(T.lplus(small).lminus(T), small, rtol=0, atol=1e-11)
    assert T.rplus(small).isclose(Transform(rm.rplus_se3(POSES, small)), atol=0)
    assert T.lplus(small).isclose(Transform(rm.lplus_se3(small, POSES)), atol=0)
    assert_allclose(OTHER.rminus(T), rm.rminus_se3(OTHER.matrix, POSES), rtol=0, atol=0)
    assert_allclose(OTHER.lminus(T), rm.lminus_se3(OTHER.matrix, POSES), rtol=0, atol=0)


def test_equality_is_exact_and_not_hashable():
    """== compares matrices exactly, isclose allows a tolerance, hashing is refused."""
    a = Transform(POSES[0])
    assert a == Transform(POSES[0].copy())
    assert a != Transform(POSES[1])
    assert a != POSES[0]
    assert not a.isclose(Transform(POSES[1]))
    assert a.isclose(Transform(POSES[0] + 1e-12))
    with pytest.raises(TypeError):
        hash(a)


def test_isclose_uses_an_absolute_tolerance_of_1e_9():
    """The default tolerance is absolute: 5e-10 apart is close, 1.5e-9 is not."""
    a = Transform(np.eye(4))
    assert a.isclose(Transform(np.eye(4) + 5e-10))
    assert not a.isclose(Transform(np.eye(4) + 1.5e-9))
    assert not a.isclose(Transform(2 * np.eye(4)))  # a relative test would accept this

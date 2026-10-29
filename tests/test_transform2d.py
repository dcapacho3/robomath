"""Robomath - Test the Transform2D class against the SE(2) functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_rotations import TOL

import robomath as rm
from robomath import Transform2D

RNG = np.random.default_rng(0)
TAUS = RNG.normal(size=(30, 3))
TAUS[:, 2] = np.linspace(-3.0, 3.0, 30)
POSES = rm.exp_vec_se2(TAUS)
T = Transform2D(POSES)
OTHER = Transform2D(np.roll(POSES, 1, axis=0))


def test_constructor_checks_shape_and_stores_a_read_only_copy():
    """Wrong shapes raise; the stored matrix is a float copy that cannot be edited."""
    with pytest.raises(ValueError, match="3, 3"):
        Transform2D(np.eye(4))
    source = np.eye(3, dtype=int)
    t = Transform2D(source)
    source[0, 2] = 9
    assert t.matrix[0, 2] == 0.0
    assert t.matrix.dtype == float
    with pytest.raises(ValueError, match="read-only"):
        t.matrix[0, 0] = 2.0
    assert Transform2D.identity().matrix.shape == (3, 3)
    assert T.shape == (30,)


def test_constructors_and_accessors_agree():
    """from_rt, from_xyt and exp build the pose the functions build."""
    r, t = rm.trans_to_rp(POSES)
    assert_allclose(Transform2D.from_rt(r, t).matrix, POSES, rtol=0, atol=0)
    assert_allclose(Transform2D.exp(TAUS).matrix, POSES, rtol=0, atol=0)
    xyt = rm.se2_to_xyt(POSES)
    assert_allclose(Transform2D.from_xyt(xyt).matrix, POSES, rtol=0, atol=TOL)
    assert_allclose(T.xyt, xyt, rtol=0, atol=0)
    assert_allclose(T.rotation, r, rtol=0, atol=0)
    assert_allclose(T.translation, t, rtol=0, atol=0)


def test_from_matrix_repairs_a_drifted_pose():
    """Noise in the rotation and a wrong bottom row come out as a true pose."""
    noisy = POSES + 1e-3 * RNG.normal(size=POSES.shape)
    fixed = Transform2D.from_matrix(noisy)
    r = fixed.rotation
    assert_allclose(
        r @ np.swapaxes(r, -1, -2),
        np.broadcast_to(np.eye(2), r.shape),
        rtol=0,
        atol=1e-12,
    )
    assert_allclose(np.linalg.det(r), 1.0, rtol=0, atol=1e-12)
    assert_allclose(fixed.matrix[:, 2], [[0, 0, 1]] * 30, rtol=0, atol=0)


def test_matmul_composes_and_inverse_undoes():
    """T_ab @ T_bc is the matrix product, and T @ T.inverse() is the identity."""
    assert_allclose((T @ OTHER).matrix, T.matrix @ OTHER.matrix, rtol=0, atol=0)
    assert isinstance(T @ OTHER, Transform2D)
    identity = Transform2D(np.broadcast_to(np.eye(3), T.matrix.shape))
    assert (T @ T.inverse()).isclose(identity, atol=1e-12)
    assert (T.inverse() @ T).isclose(identity, atol=1e-12)


def test_matmul_moves_points():
    """T @ p is R p + t for one point, a list of points, and a batch of poses."""
    one = Transform2D(POSES[0])
    p = RNG.normal(size=2)
    assert_allclose(one @ p, POSES[0, :2, :2] @ p + POSES[0, :2, 2], rtol=0, atol=TOL)
    pts = RNG.normal(size=(8, 2))
    assert_allclose(one @ pts, rm.act_se2(POSES[0], pts), rtol=0, atol=TOL)
    each = RNG.normal(size=(30, 2))
    moved = T @ each
    assert_allclose(T.inverse() @ moved, each, rtol=0, atol=1e-12)


def test_matmul_rejects_what_it_cannot_use():
    """A wrong shape, and an array on the left, raise TypeError."""
    with pytest.raises(TypeError):
        T @ np.ones(3)
    with pytest.raises(TypeError):
        np.ones(2) @ T


def test_log_adjoint_and_plus_minus():
    """The methods give what the functions give."""
    assert_allclose(T.log(), rm.log_vec_se2(POSES), rtol=0, atol=0)
    assert_allclose(T.log()[:-1], rm.log_vec_se2(POSES)[:-1], rtol=0, atol=0)
    assert_allclose(T.adjoint(), rm.adjoint_se2(POSES), rtol=0, atol=0)
    small = TAUS * 0.3
    assert_allclose(T.rplus(small).rminus(T), small, rtol=0, atol=1e-11)
    assert_allclose(T.lplus(small).lminus(T), small, rtol=0, atol=1e-11)
    assert T.rplus(small).isclose(Transform2D(rm.rplus_se2(POSES, small)), atol=0)
    assert T.lplus(small).isclose(Transform2D(rm.lplus_se2(small, POSES)), atol=0)
    assert_allclose(OTHER.rminus(T), rm.rminus_se2(OTHER.matrix, POSES), atol=0)
    assert_allclose(OTHER.lminus(T), rm.lminus_se2(OTHER.matrix, POSES), atol=0)


def test_interp_matches_the_function_and_the_exact_short_step():
    """Fractions 0 and 1 give the ends; half the step is half of tau."""
    assert T.interp(OTHER, 0.0).isclose(T, atol=1e-12)
    assert T.interp(OTHER, 1.0).isclose(OTHER, atol=1e-12)
    assert T.interp(OTHER, 0.3).isclose(
        Transform2D(rm.interp_se2(POSES, OTHER.matrix, 0.3)), atol=0
    )
    arc = Transform2D.identity().rplus([1.0, 0.0, 2.0])
    half = Transform2D.identity().interp(arc, 0.5)
    assert half.isclose(Transform2D.identity().rplus([0.5, 0.0, 1.0]), atol=1e-12)


def test_interp_goes_the_short_way_across_the_seam():
    """From 3.1 rad to -3.1 rad the middle is pi, not 0."""
    a = Transform2D.from_xyt([0.0, 0.0, 3.1])
    b = Transform2D.from_xyt([0.0, 0.0, -3.1])
    assert abs(a.interp(b, 0.5).xyt[2]) == pytest.approx(np.pi)


def test_equality_is_exact_and_not_hashable():
    """== compares matrices exactly, isclose allows a tolerance, hashing is refused."""
    a = Transform2D(POSES[0])
    assert a == Transform2D(POSES[0].copy())
    assert a != Transform2D(POSES[1])
    assert a != POSES[0]
    assert not a.isclose(Transform2D(POSES[1]))
    assert a.isclose(Transform2D(POSES[0] + 1e-12))
    with pytest.raises(TypeError):
        hash(a)

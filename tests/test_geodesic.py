"""Robomath - Test distance, interpolation and mean on SO(3) and SE(3)."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from test_rotations import TOL

import robomath as rm
from robomath import geodesic

RNG = np.random.default_rng(0)
R1 = Rotation.random(30, rng=1)
R2 = Rotation.random(30, rng=2)
M1, M2 = R1.as_matrix(), R2.as_matrix()
T = RNG.uniform(0, 1, 30)
POSES1 = rm.rp_to_trans(M1, RNG.normal(size=(30, 3)))
POSES2 = rm.rp_to_trans(M2, RNG.normal(size=(30, 3)))
# Ten rotations (and poses) within 0.3 rad of a centre, so the mean is unique.
CLOUD = rm.rplus(M1[:, None], 0.3 * RNG.normal(size=(30, 10, 3)))
POSE_CLOUD = rm.rplus_se3(POSES1[:, None], 0.3 * RNG.normal(size=(30, 10, 6)))


def test_distance_matches_scipy():
    """The angle of R1^T R2, which scipy calls the magnitude."""
    assert_allclose(
        rm.distance_so3(M1, M2), (R1.inv() * R2).magnitude(), rtol=0, atol=TOL
    )


def test_distance_properties():
    """Zero to itself, symmetric, and pi for a half turn."""
    assert_allclose(rm.distance_so3(M1, M1), 0.0, rtol=0, atol=TOL)
    assert_allclose(rm.distance_so3(M1, M2), rm.distance_so3(M2, M1), rtol=0, atol=TOL)
    assert_allclose(rm.distance_so3(np.eye(3), rm.rotx(np.pi)), np.pi, rtol=0, atol=TOL)


def test_interp_so3_matches_slerp():
    """The geodesic between rotations is the slerp of their quaternions."""
    q1, q2 = rm.matrix_to_quat(M1), rm.matrix_to_quat(M2)
    expected = rm.quat_to_matrix(rm.slerp(q1, q2, T))
    assert_allclose(rm.interp_so3(M1, M2, T), expected, rtol=0, atol=1e-12)


def test_interp_ends_and_midpoint():
    """T = 0 gives the first, t = 1 the second, and t = 1/2 is equally far."""
    assert_allclose(rm.interp_so3(M1, M2, 0.0), M1, rtol=0, atol=TOL)
    assert_allclose(rm.interp_so3(M1, M2, 1.0), M2, rtol=0, atol=1e-12)
    mid = rm.interp_so3(M1, M2, 0.5)
    assert_allclose(
        rm.distance_so3(M1, mid), rm.distance_so3(mid, M2), rtol=0, atol=1e-12
    )
    assert_allclose(rm.interp_se3(POSES1, POSES2, 0.0), POSES1, rtol=0, atol=TOL)
    assert_allclose(rm.interp_se3(POSES1, POSES2, 1.0), POSES2, rtol=0, atol=1e-12)


def test_interp_se3_moves_at_constant_rate():
    """Each step of a quarter of the way is the same tangent vector."""
    step = rm.rminus_se3(rm.interp_se3(POSES1, POSES2, 0.25), POSES1)
    assert_allclose(step * 4, rm.rminus_se3(POSES2, POSES1), rtol=0, atol=1e-12)


def test_mean_of_equal_items_and_two_items():
    """The mean of copies is the item; the mean of two is their midpoint."""
    same = np.repeat(M1[:, None], 3, axis=1)
    assert_allclose(rm.mean_so3(same), M1, rtol=0, atol=1e-12)
    two = np.stack([M1, M2], axis=1)
    assert_allclose(rm.mean_so3(two), rm.interp_so3(M1, M2, 0.5), rtol=0, atol=1e-12)
    two = np.stack([POSES1, POSES2], axis=1)
    assert_allclose(
        rm.mean_se3(two), rm.interp_se3(POSES1, POSES2, 0.5), rtol=0, atol=1e-12
    )


def test_mean_is_where_the_offsets_cancel():
    """At the mean, the weighted sum of the offsets to the items is zero."""
    w = RNG.uniform(0.5, 2, (30, 10))
    for weights in (None, w):
        mean = rm.mean_so3(CLOUD, weights)
        offsets = rm.rminus(CLOUD, mean[:, None])
        wn = np.full((30, 10), 0.1) if weights is None else w / w.sum(-1, keepdims=True)
        assert_allclose((wn[..., None] * offsets).sum(-2), 0, rtol=0, atol=1e-12)
        mean = rm.mean_se3(POSE_CLOUD, weights)
        offsets = rm.rminus_se3(POSE_CLOUD, mean[:, None])
        assert_allclose((wn[..., None] * offsets).sum(-2), 0, rtol=0, atol=1e-12)


@pytest.mark.filterwarnings("ignore:an item is")  # some clouds spread wide
def test_mean_weight_on_one_item():
    """A weight of one on one item gives that item."""
    w = np.zeros(10)
    w[3] = 1.0
    assert_allclose(rm.mean_so3(CLOUD, w), CLOUD[:, 3], rtol=0, atol=1e-12)


def test_mean_close_to_scipy_for_tight_cluster():
    """For a tight cluster the Karcher mean and scipy's chordal mean nearly agree."""
    tight = rm.rplus(M1[:, None], 0.01 * RNG.normal(size=(30, 10, 3)))
    for cloud in tight[:5]:
        expected = Rotation.from_matrix(cloud).mean().as_matrix()
        assert_allclose(rm.mean_so3(cloud), expected, rtol=0, atol=1e-5)


def test_se2_interp_and_mean():
    """SE(2) interpolation: both ends, constant speed; mean of two is the midpoint."""
    taus = RNG.normal(size=(30, 2, 3))
    a, b = rm.exp_vec_se2(taus[:, 0]), rm.exp_vec_se2(taus[:, 1])
    assert_allclose(rm.interp_se2(a, b, 0.0), a, rtol=0, atol=TOL)
    assert_allclose(rm.interp_se2(a, b, 1.0), b, rtol=0, atol=1e-12)
    step = rm.rminus_se2(rm.interp_se2(a, b, 0.25), a)
    assert_allclose(step * 4, rm.rminus_se2(b, a), rtol=0, atol=1e-12)
    two = np.stack([a, b], axis=1)
    assert_allclose(rm.mean_se2(two), rm.interp_se2(a, b, 0.5), rtol=0, atol=1e-12)


def test_mean_raises_when_it_does_not_settle(monkeypatch):
    """Running out of iterations is an error, not a silent answer."""
    monkeypatch.setattr("robomath.geodesic.MEAN_ITERATIONS", 1)
    with pytest.raises(ValueError, match="settle"):
        rm.mean_so3(CLOUD)


def test_mean_warns_when_the_items_are_far_apart():
    """Three rotations 120 degrees apart have no unique mean; the call says so."""
    third = np.array([rm.rotz(0), rm.rotz(2 * np.pi / 3), rm.rotz(-2 * np.pi / 3)])
    with pytest.warns(UserWarning, match="not be unique"):
        rm.mean_so3(third)
    poses = rm.rp_to_trans(third, np.zeros((3, 3)))
    with pytest.warns(UserWarning, match="not be unique"):
        rm.mean_se3(poses)
    planar = rm.exp_vec_se2(np.array([[0, 0, 0], [0, 0, 2.1], [0, 0, -2.1]]))
    with pytest.warns(UserWarning, match="not be unique"):
        rm.mean_se2(planar)


def test_mean_does_not_warn_for_items_80_degrees_apart():
    """Two rotations 160 degrees apart are each 80 degrees from their mean."""
    pair = np.array([rm.rotz(0.0), rm.rotz(np.radians(160))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean = rm.mean_so3(pair)
    assert_allclose(mean, rm.rotz(np.radians(80)), rtol=0, atol=1e-12)


def test_start_matches_scipy_chordal_mean():
    """The first guess for the SO(3) mean is scipy's chordal mean."""
    w = np.full((30, 10), 0.1)
    got = geodesic._start_so3(CLOUD, w)  # pylint: disable=protected-access
    for g, cloud in zip(got, CLOUD, strict=True):
        expected = Rotation.from_matrix(cloud).mean().as_matrix()
        assert_allclose(g, expected, rtol=0, atol=1e-12)


@pytest.mark.filterwarnings("ignore:an item is")  # these clouds are wide
def test_mean_does_not_depend_on_item_order():
    """Wide clouds give the same mean whatever order the items come in."""
    wide = rm.exp_vec(RNG.normal(size=(30, 8, 3)) * 0.9)
    poses = rm.exp_vec_se3(RNG.normal(size=(30, 8, 6)) * 0.9)
    planar = rm.exp_vec_se2(RNG.normal(size=(30, 8, 3)) * 0.9)
    for mean, items in (
        (rm.mean_so3, wide),
        (rm.mean_se3, poses),
        (rm.mean_se2, planar),
    ):
        assert_allclose(mean(items), mean(items[:, ::-1]), rtol=0, atol=1e-9)


def test_mean_rejects_bad_weights_and_empty_input():
    """Negative or all-zero weights and an empty set are errors."""
    with pytest.raises(ValueError, match="weights"):
        rm.mean_so3(CLOUD[0], -np.ones(10))
    with pytest.raises(ValueError, match="weights"):
        rm.mean_so3(CLOUD[0], np.zeros(10))
    with pytest.raises(ValueError, match="no items"):
        rm.mean_so3(np.zeros((0, 3, 3)))

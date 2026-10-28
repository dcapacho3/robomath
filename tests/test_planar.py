"""Robomath - Test the -pi range, the (x, y, angle) conversion and the act functions."""

import numpy as np
from numpy.testing import assert_allclose
from test_rotations import TOL

import robomath as rm

PI = np.pi
RNG = np.random.default_rng(1)


def test_minus_pi_maps_to_pi():
    """-pi and pi name one angle, and the range is (-pi, pi]."""
    assert rm.wrap_angle(-PI) == PI
    assert rm.wrap_angle(PI) == PI
    assert_allclose(rm.wrap_angle(3 * PI), PI, rtol=0, atol=1e-12)
    assert rm.log_so2(rm.exp_so2(-PI)) == PI
    assert_allclose(rm.wrap_angle(-PI + 1e-9), -PI + 1e-9, rtol=0, atol=1e-15)
    assert_allclose(rm.wrap_angle(PI + 1e-9), -PI + 1e-9, rtol=0, atol=1e-8)
    both = rm.wrap_angle(np.array([[-PI, PI], [0.5, -0.5]]))
    assert_allclose(both, [[PI, PI], [0.5, -0.5]], rtol=0, atol=TOL)


def test_log_vec_se2_keeps_the_range_for_one_pose_and_a_batch():
    """The single-pose path and the batch path agree at a half turn."""
    one = rm.log_vec_se2(rm.xyt_to_se2([1.0, 2.0, -PI]))
    assert one[2] == PI
    many = rm.log_vec_se2(rm.xyt_to_se2([[1.0, 2.0, -PI], [1.0, 2.0, PI]]))
    assert_allclose(many, [one, one], rtol=0, atol=TOL)


def test_xyt_quarter_turn():
    """A quarter turn at (1, 2) is [[0, -1, 1], [1, 0, 2], [0, 0, 1]]."""
    m = rm.xyt_to_se2([1.0, 2.0, PI / 2])
    assert_allclose(m, [[0, -1, 1], [1, 0, 2], [0, 0, 1]], rtol=0, atol=TOL)


def test_xyt_round_trip_and_batch():
    """se2_to_xyt undoes xyt_to_se2; -pi comes back as pi; batches keep their shape."""
    xyt = np.column_stack([RNG.normal(size=(4, 2)), RNG.uniform(-3, 3, 4)])
    assert rm.xyt_to_se2(xyt).shape == (4, 3, 3)
    assert_allclose(rm.se2_to_xyt(rm.xyt_to_se2(xyt)), xyt, rtol=0, atol=TOL)
    back = rm.se2_to_xyt(rm.xyt_to_se2([0.0, 1.0, -PI]))
    assert_allclose(back, [0.0, 1.0, PI], rtol=0, atol=TOL)


def test_xyt_is_not_a_tangent_vector():
    """exp_vec_se2 of (x, y, angle) matches xyt_to_se2 at angle 0 and not at 0.5."""
    flat, turned = np.array([1.0, 2.0, 0.0]), np.array([1.0, 2.0, 0.5])
    assert_allclose(rm.exp_vec_se2(flat), rm.xyt_to_se2(flat), rtol=0, atol=TOL)
    assert not np.allclose(rm.exp_vec_se2(turned), rm.xyt_to_se2(turned))


def test_act_se2():
    """Identity moves nothing; a quarter turn at (1, 2) moves (1, 0) to (1, 3)."""
    p = RNG.normal(size=(5, 2))
    assert_allclose(rm.act_se2(np.eye(3), p), p, rtol=0, atol=0)
    quarter = rm.xyt_to_se2([1.0, 2.0, PI / 2])
    assert_allclose(rm.act_se2(quarter, [[1.0, 0.0]]), [[1.0, 3.0]], rtol=0, atol=TOL)
    a, b = rm.xyt_to_se2([1, 2, 0.4]), rm.xyt_to_se2([-1, 0.5, 2.0])
    assert_allclose(
        rm.act_se2(a @ b, p), rm.act_se2(a, rm.act_se2(b, p)), rtol=0, atol=TOL
    )
    poses = rm.xyt_to_se2(RNG.normal(size=(3, 3)))
    assert rm.act_se2(poses, p).shape == (3, 5, 2)  # one point set, three poses
    assert rm.act_se2(poses, RNG.normal(size=(3, 5, 2))).shape == (3, 5, 2)
    assert rm.act_se2(poses[0], p).shape == (5, 2)  # one pose, many points


def test_act_se3_matches_transform_and_act_se2():
    """act_se3 equals Transform @, and a planar SE(3) pose acts like its SE(2) one."""
    m = rm.exp_vec_se3(RNG.normal(size=(3, 6)))
    p = RNG.normal(size=(5, 3))
    assert_allclose(rm.act_se3(m, p), rm.Transform(m[:, None]) @ p, rtol=0, atol=TOL)
    xyt = np.array([1.0, -2.0, 0.7])
    r, t = rm.rotz(xyt[2]), np.array([xyt[0], xyt[1], 0.5])
    flat = rm.act_se3(rm.rp_to_trans(r, t), np.c_[p[:, :2], np.zeros(5)])
    assert_allclose(flat[:, :2], rm.act_se2(rm.xyt_to_se2(xyt), p[:, :2]), atol=TOL)


def test_act_sim3_scales_then_moves():
    """A similarity moves a point as s R p + t."""
    s, r, t = 2.0, rm.rotz(0.3), np.array([1.0, 2.0, 3.0])
    p = RNG.normal(size=(5, 3))
    assert_allclose(
        rm.act_sim3(rm.srt_to_sim3(s, r, t), p), s * p @ r.T + t, rtol=0, atol=TOL
    )

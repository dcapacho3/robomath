"""Robomath - Test axis-angle and quaternion conversions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from test_rotations import TOL

from robomath import (
    axis_angle_to_matrix,
    axis_angle_to_quat,
    matrix_to_axis_angle,
    matrix_to_quat,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_axis_angle,
    quat_to_matrix,
    rotx,
    roty,
    rotz,
    slerp,
)

# scipy quaternions are xyzw too. as_quat() can return either sign, so
# rotations are compared as matrices.
RANDOM_ROTATIONS = Rotation.random(100, rng=1)
RNG = np.random.default_rng(0)
AXES = RNG.normal(size=(30, 3))
AXES /= np.linalg.norm(AXES, axis=1, keepdims=True)
Q1, Q2 = RANDOM_ROTATIONS[0].as_quat(), RANDOM_ROTATIONS[1].as_quat()


@pytest.mark.parametrize("angle", [0.0, 0.3, 1.5, np.pi])
def test_axis_angle_axes_are_rotxyz(angle):
    """A coordinate axis reproduces rotx, roty, rotz."""
    for axis, rot in zip(np.eye(3), (rotx, roty, rotz), strict=True):
        assert_allclose(axis_angle_to_matrix(axis, angle), rot(angle), atol=TOL)


@pytest.mark.parametrize("axis", AXES)
def test_axis_angle_matches_scipy(axis):
    """Both directions against scipy's rotation vector."""
    rotvec = axis * 1.1
    expected = Rotation.from_rotvec(rotvec).as_matrix()
    assert_allclose(axis_angle_to_matrix(axis, 1.1), expected, atol=TOL)
    got_axis, got_angle = matrix_to_axis_angle(expected)
    assert_allclose(got_axis * got_angle, rotvec, atol=TOL)


def test_axis_angle_round_trip():
    """Matrix -> axis-angle -> matrix. Compare matrices: the axis flips at pi."""
    for m in RANDOM_ROTATIONS.as_matrix():
        axis, angle = matrix_to_axis_angle(m)
        assert_allclose(axis_angle_to_matrix(axis, angle), m, atol=TOL)


@pytest.mark.parametrize("angle", [1e-12, 1e-9, 1e-7, 1e-5, 1e-3])
def test_small_angles(angle):
    """Tiny rotations keep their angle (the book's arccos would lose it)."""
    axis = AXES[0]
    got_axis, got_angle = matrix_to_axis_angle(axis_angle_to_matrix(axis, angle))
    assert got_angle == pytest.approx(angle, rel=1e-6)
    assert_allclose(got_axis, axis, atol=1e-3)


@pytest.mark.parametrize("gap", [1e-3, 1e-5, 1e-7, 1e-8, 1e-9, 1e-12, 0.0])
def test_near_pi(gap):
    """Angles on both sides of SIN_TOL rebuild the same matrix."""
    for axis in AXES[:10]:
        m = axis_angle_to_matrix(axis, np.pi - gap)
        got_axis, got_angle = matrix_to_axis_angle(m)
        assert_allclose(axis_angle_to_matrix(got_axis, got_angle), m, atol=1e-7)


def test_angle_zero_and_pi():
    """Identity gives angle 0; half turns about x, y, z give angle pi."""
    assert matrix_to_axis_angle(np.eye(3))[1] == 0.0
    for axis in np.eye(3):
        got_axis, got_angle = matrix_to_axis_angle(axis_angle_to_matrix(axis, np.pi))
        assert got_angle == pytest.approx(np.pi)
        assert abs(np.dot(got_axis, axis)) == pytest.approx(1.0)  # sign is free


def test_quat_multiply_matches_matrix_product():
    """Checks the xyzw order and the multiply: R(q1 q2) = R(q1) R(q2)."""
    assert_allclose(
        quat_to_matrix(quat_multiply(Q1, Q2)),
        quat_to_matrix(Q1) @ quat_to_matrix(Q2),
        atol=TOL,
    )


def test_quarter_turns_make_half_turn():
    """Two 90-degree turns about z give (0, 0, 1, 0)."""
    q = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    assert_allclose(quat_multiply(q, q), [0, 0, 1, 0], atol=TOL)


def test_quat_not_commutative():
    """Turns about different axes do not commute."""
    qx = axis_angle_to_quat(np.array([1.0, 0, 0]), 0.5)
    qy = axis_angle_to_quat(np.array([0, 1.0, 0]), 0.5)
    assert not np.allclose(quat_multiply(qx, qy), quat_multiply(qy, qx))


def test_quat_conjugate_is_inverse():
    """Q * conj(q) is the identity quaternion."""
    assert_allclose(quat_multiply(Q1, quat_conjugate(Q1)), [0, 0, 0, 1], atol=TOL)


def test_double_cover():
    """Q and -q give the same matrix."""
    assert_allclose(quat_to_matrix(Q1), quat_to_matrix(-Q1), atol=TOL)


def test_quat_to_matrix_matches_scipy():
    """Against Rotation.as_matrix on 100 random rotations."""
    for rot in RANDOM_ROTATIONS:
        assert_allclose(quat_to_matrix(rot.as_quat()), rot.as_matrix(), atol=TOL)


def test_quat_round_trip():
    """Quat -> matrix -> quat; scalar >= 0 on both sides, so compare directly."""
    for rot in RANDOM_ROTATIONS:
        q = rot.as_quat()
        q = q if q[3] >= 0 else -q
        assert_allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-7)


def test_quat_axis_angle_match_scipy():
    """axis_angle_to_quat and quat_to_axis_angle against scipy's rotvec."""
    for axis in AXES:
        q = axis_angle_to_quat(axis, 0.9)
        assert_allclose(q, Rotation.from_rotvec(axis * 0.9).as_quat(), atol=TOL)
        got_axis, got_angle = quat_to_axis_angle(-q)  # negative scalar is flipped
        assert_allclose(got_axis, axis, atol=TOL)
        assert got_angle == pytest.approx(0.9)
    assert quat_to_axis_angle(np.array([0, 0, 0, 1.0]))[1] == 0.0


def test_normalize():
    """Unit length after; the zero quaternion raises."""
    assert np.linalg.norm(quat_normalize(np.array([1.0, 2, 3, 4]))) == pytest.approx(1)
    with pytest.raises(ValueError, match="zero quaternion"):
        quat_normalize(np.zeros(4))


def test_slerp_endpoints_and_midpoint():
    """T = 0 and 1 give q0 and q1 (up to sign); every t matches scipy."""
    ref = Slerp([0, 1], Rotation.from_quat([Q1, Q2]))
    for t in (0.0, 0.25, 0.5, 1.0):
        assert_allclose(
            quat_to_matrix(slerp(Q1, Q2, t)), ref([t]).as_matrix()[0], atol=TOL
        )


def test_slerp_shortest_path():
    """q1 and -q1 give the same rotation."""
    assert_allclose(
        quat_to_matrix(slerp(Q1, Q2, 0.4)),
        quat_to_matrix(slerp(Q1, -Q2, 0.4)),
        atol=TOL,
    )


def test_slerp_nearly_equal_quats():
    """The lerp branch returns a unit quaternion, no NaN."""
    q = slerp(Q1, Q1, 0.5)
    assert np.isfinite(q).all()
    assert np.linalg.norm(q) == pytest.approx(1)
    assert_allclose(quat_to_matrix(q), quat_to_matrix(Q1), atol=TOL)


def test_slerp_constant_angular_speed():
    """The angle from q0 grows linearly with t."""
    total = matrix_to_axis_angle(quat_to_matrix(Q1).T @ quat_to_matrix(Q2))[1]
    for t in (0.1, 0.5, 0.9):
        step = matrix_to_axis_angle(
            quat_to_matrix(Q1).T @ quat_to_matrix(slerp(Q1, Q2, t))
        )[1]
        assert step == pytest.approx(t * total)

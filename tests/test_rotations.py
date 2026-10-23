"""Robomath - Test all rotations are correct in known cases."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from robomath import inverse, rotx, roty, rotz

AXES = [rotx, roty, rotz]
ANGLES = [0.0, 0.3, np.pi / 2, np.pi, -1.2]
TOL = 1e-12


@pytest.mark.parametrize("r", AXES)
def test_zero_angle_is_identity(r):
    """A zero angle gives the identity matrix."""
    assert_allclose(r(0.0), np.eye(3), rtol=0, atol=TOL)


@pytest.mark.parametrize(
    "rot, start, end",
    [
        (rotx, [0, 1, 0], [0, 0, 1]),
        (roty, [0, 0, 1], [1, 0, 0]),
        (rotz, [1, 0, 0], [0, 1, 0]),
    ],
)
def test_quarter_turn(rot, start, end):
    """A +90 degree turn moves one axis onto the next.

    Rotating about z moves x to y, about x moves y to z, and about y moves z to x.
    """
    assert_allclose(rot(np.pi / 2) @ start, end, rtol=0, atol=TOL)


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("r", AXES)
def test_orthogonal_with_det_one(r, theta):
    """A rotation matrix is orthogonal and has determinant +1."""
    m = r(theta)
    assert_allclose(m @ m.T, np.eye(3), rtol=0, atol=TOL)
    assert_allclose(np.linalg.det(m), 1.0, rtol=0, atol=TOL)


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("r", AXES)
def test_inverse(r, theta):
    """A matrix times its inverse gives the identity, and matches np.linalg.inv."""
    m = r(theta)
    assert_allclose(m @ inverse(m), np.eye(3), rtol=0, atol=TOL)
    assert_allclose(inverse(m), np.linalg.inv(m), rtol=0, atol=TOL)


def test_same_axis_angles_add():
    """Rotations about the same axis add their angles."""
    assert_allclose(rotx(0.4) @ rotx(0.5), rotx(0.9), rtol=0, atol=TOL)
    assert_allclose(roty(0.4) @ roty(0.5), roty(0.9), rtol=0, atol=TOL)
    assert_allclose(rotz(0.4) @ rotz(0.5), rotz(0.9), rtol=0, atol=TOL)


def test_order_matters():
    """Rotations about different axes do not commute."""
    a = rotx(0.4) @ rotz(0.3)
    b = rotz(0.3) @ rotx(0.4)
    assert not np.allclose(a, b, atol=TOL)

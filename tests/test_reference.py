"""Check robomath against independent libraries: scipy and modern_robotics."""

import modern_robotics as mr
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from test_rotations import ANGLES, TOL

from robomath import inverse, rotx, roty, rotz

# (robomath function, scipy axis letter)
CASES = [(rotx, "x"), (roty, "y"), (rotz, "z")]
UNIT = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}

# Fixed seed: a failure must be reproducible.
RANDOM_ANGLES = np.random.default_rng(0).uniform(-2 * np.pi, 2 * np.pi, 20)
RANDOM_TRIPLES = np.random.default_rng(1).uniform(-2 * np.pi, 2 * np.pi, (10, 3))


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("rot, axis", CASES)
def test_matches_scipy(rot, axis, theta):
    """Each rotation matches scipy's matrix for the same axis and angle."""
    expected = Rotation.from_euler(axis, theta).as_matrix()
    assert_allclose(rot(theta), expected, rtol=0, atol=TOL)


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("rot, axis", CASES)
def test_matches_modern_robotics(rot, axis, theta):
    """Each rotation matches the matrix exponential of the axis times the angle."""
    omega = theta * np.array(UNIT[axis])
    expected = mr.MatrixExp3(mr.VecToso3(omega))
    assert_allclose(rot(theta), expected, rtol=0, atol=TOL)


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("rot", [rotx, roty, rotz])
def test_inverse_matches_modern_robotics(rot, theta):
    """The inverse matches mr.RotInv."""
    assert_allclose(inverse(rot(theta)), mr.RotInv(rot(theta)), rtol=0, atol=TOL)


@pytest.mark.parametrize("theta", RANDOM_ANGLES)
@pytest.mark.parametrize("rot, axis", CASES)
def test_random_angles(rot, axis, theta):
    """Random angles, including large ones, agree with both libraries."""
    m = rot(theta)
    omega = theta * np.array(UNIT[axis])
    assert_allclose(m, Rotation.from_euler(axis, theta).as_matrix(), rtol=0, atol=TOL)
    assert_allclose(m, mr.MatrixExp3(mr.VecToso3(omega)), rtol=0, atol=TOL)


@pytest.mark.parametrize("angles", RANDOM_TRIPLES)
def test_chain_matches_scipy(angles):
    """Rz(a) @ Ry(b) @ Rx(c) equals scipy's intrinsic ZYX rotation."""
    a, b, c = angles
    expected = Rotation.from_euler("ZYX", [a, b, c]).as_matrix()
    assert_allclose(rotz(a) @ roty(b) @ rotx(c), expected, rtol=0, atol=TOL)

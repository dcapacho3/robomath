"""Robomath - The Lie group SO(2) of planar rotations.

A rotation is an angle theta in radians, or the 2x2 matrix [[c, -s], [s, c]].
Every function accepts leading batch dimensions.
"""

import numpy as np


def exp_so2(theta: np.ndarray) -> np.ndarray:
    """Rotation matrix of an angle.

    Args:
        theta (np.ndarray): Angles in radians, shape (...).

    Returns:
        np.ndarray: The matrices [[cos, -sin], [sin, cos]], shape (..., 2, 2).
    """
    theta = np.asarray(theta, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)], axis=-2)


def log_so2(r: np.ndarray) -> np.ndarray:
    """Angle of a rotation matrix, in (-pi, pi].

    Args:
        r (np.ndarray): Rotation matrices, shape (..., 2, 2).

    Returns:
        np.ndarray: The angles, shape (...).
    """
    r = np.asarray(r, dtype=float)
    return np.arctan2(r[..., 1, 0], r[..., 0, 0])


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """Angle equal to theta modulo 2 pi, in (-pi, pi]."""
    theta = np.asarray(theta, dtype=float)
    return np.arctan2(np.sin(theta), np.cos(theta))

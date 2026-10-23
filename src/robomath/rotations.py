"""Robomath - Rotation matrices about the x, y and z axes.

Every function accepts leading batch dimensions: an angle array of shape (...)
gives matrices of shape (..., 3, 3). Results are float64.
"""

import numpy as np


def rotz(theta: float) -> np.ndarray:
    """Rotate by theta radians about z.

    Args:
        theta (float): Angle in radians, scalar or shape (...). Positive turns
            x toward y.

    Returns:
        np.ndarray: The rotation matrix, shape (..., 3, 3).
    """
    theta = np.asarray(theta, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    r = np.zeros(theta.shape + (3, 3))
    r[..., 0, 0] = c
    r[..., 0, 1] = -s
    r[..., 1, 0] = s
    r[..., 1, 1] = c
    r[..., 2, 2] = 1
    return r


def rotx(theta: float) -> np.ndarray:
    """Rotate by theta radians about x.

    Args:
        theta (float): Angle in radians, scalar or shape (...). Positive turns
            y toward z.

    Returns:
        np.ndarray: The rotation matrix, shape (..., 3, 3).
    """
    theta = np.asarray(theta, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    r = np.zeros(theta.shape + (3, 3))
    r[..., 0, 0] = 1
    r[..., 1, 1] = c
    r[..., 1, 2] = -s
    r[..., 2, 1] = s
    r[..., 2, 2] = c
    return r


def roty(theta: float) -> np.ndarray:
    """Rotate by theta radians about y.

    Args:
        theta (float): Angle in radians, scalar or shape (...). Positive turns
            z toward x.

    Returns:
        np.ndarray: The rotation matrix, shape (..., 3, 3).
    """
    theta = np.asarray(theta, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    r = np.zeros(theta.shape + (3, 3))
    r[..., 0, 0] = c
    r[..., 0, 2] = s
    r[..., 1, 1] = 1
    r[..., 2, 0] = -s
    r[..., 2, 2] = c
    return r


def inverse(r: np.ndarray) -> np.ndarray:
    """Inverse of a rotation matrix. For a rotation, the transpose is the inverse.

    Args:
        r (np.ndarray): Rotation matrices, shape (..., 3, 3).

    Returns:
        np.ndarray: The transpose of the last two axes of r (a view, not a copy).
    """
    return np.swapaxes(r, -1, -2)

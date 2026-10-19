"""Robomath - Rotation matrices about the x, y and z axes."""

import numpy as np


def rotz(theta: float) -> np.ndarray:
    """Rotate by theta radians about z.

    Args:
        theta (float): Angle in radians. Positive turns x toward y.

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )


def rotx(theta: float) -> np.ndarray:
    """Rotate by theta radians about x.

    Args:
        theta (float): Angle in radians. Positive turns y toward z.

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]
    )


def roty(theta: float) -> np.ndarray:
    """Rotate by theta radians about y.

    Args:
        theta (float): Angle in radians. Positive turns z toward x.

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ]
    )


def inverse(r: np.ndarray) -> np.ndarray:
    """Inverse of a rotation matrix. For a rotation, the transpose is the inverse.

    Args:
        r (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: The transpose of r.
    """
    return r.T

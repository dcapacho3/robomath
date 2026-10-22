"""Robomath - ZYX Euler angles."""

import numpy as np

GIMBAL_TOL = 1e-15


def euler_zyx_to_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Rotation matrix for ZYX Euler angles.

    Args:
        alpha (float): Angle about z, in radians.
        beta (float): Angle about y, in radians.
        gamma (float): Angle about x, in radians.

    Returns:
        np.ndarray: The 3x3 rotation matrix, R = Rz(alpha) Ry(beta) Rx(gamma).
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    r11 = ca * cb
    r12 = ca * sb * sg - sa * cg
    r13 = ca * sb * cg + sa * sg
    r21 = sa * cb
    r22 = sa * sb * sg + ca * cg
    r23 = sa * sb * cg - ca * sg
    r31 = -sb
    r32 = cb * sg
    r33 = cb * cg
    return np.array(
        [
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33],
        ]
    )


def matrix_to_euler_zyx(r: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles of a rotation matrix.

    At gimbal lock (beta = +-pi/2) only one combination of alpha and gamma is
    fixed by the matrix, so alpha is set to 0 and gamma takes the rest.

    Args:
        r (np.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple[float, float, float]: (alpha, beta, gamma) in radians, with
            alpha and gamma in (-pi, pi] and beta in [-pi/2, pi/2].
    """
    [r11, r12, r13] = r[0]
    [r21, r22, r23] = r[1]
    [r31, r32, r33] = r[2]

    if abs(r31 + 1) < GIMBAL_TOL:
        beta = np.pi / 2
        alpha = 0.0
        gamma = np.arctan2(r12, r22)
    elif abs(r31 - 1) < GIMBAL_TOL:
        beta = -np.pi / 2
        alpha = 0.0
        gamma = -np.arctan2(r12, r22)
    else:
        beta = np.arctan2(-r31, np.hypot(r11, r21))
        alpha = np.arctan2(r21, r11)
        gamma = np.arctan2(r32, r33)

    return alpha, beta, gamma

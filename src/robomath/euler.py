"""Robomath - ZYX Euler angles.

Every function accepts leading batch dimensions.
"""

import math

import numpy as np

GIMBAL_TOL = 1e-15


def euler_zyx_to_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Rotation matrix for ZYX Euler angles.

    Args:
        alpha (float): Angle about z, in radians, scalar or shape (...).
        beta (float): Angle about y, in radians; broadcasts with alpha.
        gamma (float): Angle about x, in radians; broadcasts with alpha.

    Returns:
        np.ndarray: The rotation matrix, shape (..., 3, 3),
            R = Rz(alpha) Ry(beta) Rx(gamma).
    """
    alpha, beta, gamma = (np.asarray(a, dtype=float) for a in (alpha, beta, gamma))
    if alpha.ndim == beta.ndim == gamma.ndim == 0:
        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta), math.sin(beta)
        cg, sg = math.cos(gamma), math.sin(gamma)
        return np.array(
            [
                [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
                [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
                [-sb, cb * sg, cb * cg],
            ]
        )
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    r = np.empty(np.broadcast_shapes(alpha.shape, beta.shape, gamma.shape) + (3, 3))
    r[..., 0, 0] = ca * cb
    r[..., 0, 1] = ca * sb * sg - sa * cg
    r[..., 0, 2] = ca * sb * cg + sa * sg
    r[..., 1, 0] = sa * cb
    r[..., 1, 1] = sa * sb * sg + ca * cg
    r[..., 1, 2] = sa * sb * cg - ca * sg
    r[..., 2, 0] = -sb
    r[..., 2, 1] = cb * sg
    r[..., 2, 2] = cb * cg
    return r


def matrix_to_euler_zyx(r: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles of a rotation matrix.

    At gimbal lock (beta = +-pi/2) only one combination of alpha and gamma is
    fixed by the matrix, so alpha is set to 0 and gamma takes the rest.

    Args:
        r (np.ndarray): Rotation matrices, shape (..., 3, 3).

    Returns:
        tuple[float, float, float]: (alpha, beta, gamma) in radians, each of
            shape (...), with alpha and gamma in (-pi, pi] and beta in
            [-pi/2, pi/2].
    """
    r = np.asarray(r, dtype=float)
    if r.ndim == 2:
        (r11, r12, _), (r21, r22, _), (r31, r32, r33) = r.tolist()
        if abs(r31 + 1) < GIMBAL_TOL:
            return 0.0, math.pi / 2, math.atan2(r12, r22)
        if abs(r31 - 1) < GIMBAL_TOL:
            return 0.0, -math.pi / 2, -math.atan2(r12, r22)
        return (
            math.atan2(r21, r11),
            math.atan2(-r31, math.hypot(r11, r21)),
            math.atan2(r32, r33),
        )
    r11, r12 = r[..., 0, 0], r[..., 0, 1]
    r21, r22 = r[..., 1, 0], r[..., 1, 1]
    r31, r32, r33 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]

    # Both gimbal-lock cases and the general one are computed, then chosen by
    # mask, so a batch can mix them.
    lock_up = np.abs(r31 + 1) < GIMBAL_TOL
    lock_down = np.abs(r31 - 1) < GIMBAL_TOL
    locked_gamma = np.arctan2(r12, r22)

    beta = np.where(
        lock_up,
        np.pi / 2,
        np.where(lock_down, -np.pi / 2, np.arctan2(-r31, np.hypot(r11, r21))),
    )
    alpha = np.where(lock_up | lock_down, 0.0, np.arctan2(r21, r11))
    gamma = np.where(
        lock_up,
        locked_gamma,
        np.where(lock_down, -locked_gamma, np.arctan2(r32, r33)),
    )
    return alpha[()], beta[()], gamma[()]

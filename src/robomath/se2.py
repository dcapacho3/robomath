"""Robomath - The Lie group SE(2) of planar rigid motions.

A pose is a 3x3 matrix M = [[R, t], [0, 1]] with R a 2x2 rotation. A tangent
vector is tau = (rho, theta) = (x, y, theta): translation first, then the angle.
The formulas are those of SE(3) with the rotation vector reduced to one number.
The pose helpers rp_to_trans, trans_to_rp and trans_inv in se3.py also
work on 3x3 poses. Every function accepts leading batch dimensions.
"""

import math

import numpy as np

from .se3 import trans_inv
from .so2 import log_so2


def _v_coefficients(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The V of SE(2) is a I + b J, with J the +90 degree turn.

    a = sin(t)/t and b = (1 - cos(t))/t = 2 sin(t/2)^2 / t, with a = 1 and b = 0 at
    t = 0. The formulas match _v_scalars operation for operation, so a batch gives
    the same bits as a loop over single poses.
    """
    zero = theta == 0.0
    safe = np.where(zero, 1.0, theta)
    half = np.sin(theta / 2)
    return (
        np.where(zero, 1.0, np.sin(theta) / safe),
        np.where(zero, 0.0, 2 * half * half / safe),
    )


def _v_scalars(theta: float) -> tuple[float, float]:
    """The a and b of _v_coefficients for one angle, in plain floats."""
    if theta == 0.0:
        return 1.0, 0.0
    half = math.sin(theta / 2)  # half * half, as x ** 2 rounds differently in 1 in 1000
    return math.sin(theta) / theta, 2 * half * half / theta


def xyt_to_se2(xyt: np.ndarray) -> np.ndarray:
    """Pose matrix of a planar pose (x, y, angle), shape (..., 3) -> (..., 3, 3).

    The numbers are the coordinates of the pose, not a tangent vector: this is not
    exp_vec_se2 of the same three numbers. The two agree only when the angle is 0.
    """
    xyt = np.asarray(xyt, dtype=float)
    if xyt.ndim == 1:  # one pose: plain floats are several times faster than arrays
        x, y, theta = xyt.tolist()
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])
    theta = xyt[..., 2]
    c, s = np.cos(theta), np.sin(theta)
    out = np.zeros(xyt.shape[:-1] + (3, 3))
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    out[..., :2, 2] = xyt[..., :2]
    out[..., 2, 2] = 1.0
    return out


def se2_to_xyt(m: np.ndarray) -> np.ndarray:
    """Planar pose (x, y, angle) of a pose matrix, angle in (-pi, pi].

    Shape (..., 3, 3) -> (..., 3). The inverse of xyt_to_se2, and not log_vec_se2.
    """
    m = np.asarray(m)
    if m.ndim == 2:  # one pose, in plain floats
        (c, _, x), (s, _, y) = m[0].tolist(), m[1].tolist()
        theta = math.atan2(s, c)
        return np.array([x, y, math.pi if theta == -math.pi else theta])
    out = np.empty(m.shape[:-2] + (3,))
    out[..., :2] = m[..., :2, 2]
    out[..., 2] = log_so2(m[..., :2, :2])
    return out


def act_se2(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Move points by poses: p_a = R p_b + t. Shapes (..., 3, 3), (..., N, 2).

    The points need an N axis: one point has shape (1, 2).
    """
    m = np.asarray(m, dtype=float)
    return points @ np.swapaxes(m[..., :2, :2], -1, -2) + m[..., None, :2, 2]


def hat_se2(tau: np.ndarray) -> np.ndarray:
    """3x3 matrix [[0, -theta, x], [theta, 0, y], [0, 0, 0]] of a tangent vector."""
    tau = np.asarray(tau, dtype=float)
    m = np.zeros(tau.shape[:-1] + (3, 3))
    m[..., 0, 1] = -tau[..., 2]
    m[..., 1, 0] = tau[..., 2]
    m[..., :2, 2] = tau[..., :2]
    return m


def vee_se2(m: np.ndarray) -> np.ndarray:
    """Tangent vector (x, y, theta) of a 3x3 se(2) matrix. Inverse of hat_se2."""
    m = np.asarray(m, dtype=float)
    return np.concatenate([m[..., :2, 2], m[..., 1, 0, None]], axis=-1)


def exp_vec_se2(tau: np.ndarray) -> np.ndarray:
    """Pose of a tangent vector: [[R(theta), V(theta) rho], [0, 1]], (..., 3, 3)."""
    tau = np.asarray(tau, dtype=float)
    if tau.ndim == 1:  # one pose: plain floats are several times faster than arrays
        x, y, theta = tau.tolist()
        a, b = _v_scalars(theta)
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s, a * x - b * y], [s, c, b * x + a * y], [0, 0, 1.0]])
    x, y, theta = tau[..., 0], tau[..., 1], tau[..., 2]
    a, b = _v_coefficients(theta)
    c, s = np.cos(theta), np.sin(theta)
    out = np.zeros(tau.shape[:-1] + (3, 3))
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    out[..., 0, 2] = a * x - b * y
    out[..., 1, 2] = b * x + a * y
    out[..., 2, 2] = 1.0
    return out


def log_vec_se2(m: np.ndarray) -> np.ndarray:
    """Tangent vector (x, y, theta) of a pose, with theta in (-pi, pi]."""
    m = np.asarray(m, dtype=float)
    if m.ndim == 2:
        (c, _, tx), (s, _, ty) = m[0].tolist(), m[1].tolist()
        theta = math.atan2(s, c)
        if theta == -math.pi:  # keep the range (-pi, pi], as log_so2 does
            theta = math.pi
        a, b = _v_scalars(theta)
        den = a * a + b * b
        return np.array([(a * tx + b * ty) / den, (a * ty - b * tx) / den, theta])
    theta = log_so2(m[..., :2, :2])
    a, b = _v_coefficients(theta)
    den = a * a + b * b
    tx, ty = m[..., 0, 2], m[..., 1, 2]
    out = np.empty(m.shape[:-2] + (3,))
    # V^-1 = (a I - b J) / (a^2 + b^2)
    out[..., 0] = (a * tx + b * ty) / den
    out[..., 1] = (a * ty - b * tx) / den
    out[..., 2] = theta
    return out


def adjoint_se2(m: np.ndarray) -> np.ndarray:
    """3x3 adjoint [[R, (t_y, -t_x)], [0, 1]] of a pose, for (x, y, theta)."""
    m = np.asarray(m, dtype=float)
    out = np.zeros(m.shape[:-2] + (3, 3))
    out[..., :2, :2] = m[..., :2, :2]
    out[..., 0, 2] = m[..., 1, 2]
    out[..., 1, 2] = -m[..., 0, 2]
    out[..., 2, 2] = 1.0
    return out


def rplus_se2(m: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Right plus M (+) tau: move pose M by tau in its own frame."""
    return m @ exp_vec_se2(tau)


def rminus_se2(m2: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Right minus M2 (-) M1: the local tangent vector from M1 to M2."""
    return log_vec_se2(trans_inv(m1) @ m2)


def lplus_se2(tau: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Left plus tau (+) M: move pose M by tau in the global frame."""
    return exp_vec_se2(tau) @ m


def lminus_se2(m2: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Left minus M2 (-) M1: the global tangent vector from M1 to M2."""
    return log_vec_se2(m2 @ trans_inv(m1))

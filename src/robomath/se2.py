"""Robomath - The Lie group SE(2) of planar rigid motions.

A pose is a 3x3 matrix M = [[R, t], [0, 1]] with R a 2x2 rotation. A tangent
vector is tau = (rho, theta) = (x, y, theta): translation first, then the angle.
The formulas are those of SE(3) with the rotation vector reduced to one number.
The pose helpers rp_to_trans, trans_to_rp and trans_inv in se3.py also
work on 3x3 poses. Every function accepts leading batch dimensions.
"""

import math

import numpy as np

from .se3 import rp_to_trans, trans_inv, trans_to_rp
from .so2 import exp_so2, log_so2


def _perp(v: np.ndarray) -> np.ndarray:
    """Rotate vectors by +90 degrees: (x, y) -> (-y, x). Shape (..., 2)."""
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def _v_coefficients(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The V of SE(2) is a I + b J, with J the +90 degree turn.

    a = sin(t)/t and b = (1 - cos(t))/t, written with sinc so t = 0 needs no case.
    """
    a = np.sinc(theta / np.pi)
    b = theta / 2 * np.sinc(theta / (2 * np.pi)) ** 2
    return a, b


def _v_scalars(theta: float) -> tuple[float, float]:
    """The a and b of _v_coefficients for one angle, in plain floats."""
    if theta == 0.0:
        return 1.0, 0.0
    return math.sin(theta) / theta, 2 * math.sin(theta / 2) ** 2 / theta


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
    rho, theta = tau[..., :2], tau[..., 2]
    a, b = _v_coefficients(theta)
    t = a[..., None] * rho + b[..., None] * _perp(rho)
    return rp_to_trans(exp_so2(theta), t)


def log_vec_se2(m: np.ndarray) -> np.ndarray:
    """Tangent vector (x, y, theta) of a pose, with theta in (-pi, pi]."""
    m = np.asarray(m, dtype=float)
    if m.ndim == 2:
        (c, _, tx), (s, _, ty) = m[0].tolist(), m[1].tolist()
        theta = math.atan2(s, c)
        a, b = _v_scalars(theta)
        den = a * a + b * b
        return np.array([(a * tx + b * ty) / den, (a * ty - b * tx) / den, theta])
    r, t = trans_to_rp(m)
    theta = log_so2(r)
    a, b = _v_coefficients(theta)
    # V^-1 = (a I - b J) / (a^2 + b^2), and a^2 + b^2 = sinc(t / 2 pi)^2.
    rho = (a[..., None] * t - b[..., None] * _perp(t)) / (
        np.sinc(theta / (2 * np.pi)) ** 2
    )[..., None]
    return np.concatenate([rho, theta[..., None]], axis=-1)


def adjoint_se2(m: np.ndarray) -> np.ndarray:
    """3x3 adjoint [[R, (t_y, -t_x)], [0, 1]] of a pose, for (x, y, theta)."""
    r, t = trans_to_rp(m)
    out = np.zeros(r.shape[:-2] + (3, 3))
    out[..., :2, :2] = r
    out[..., 0, 2] = t[..., 1]
    out[..., 1, 2] = -t[..., 0]
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

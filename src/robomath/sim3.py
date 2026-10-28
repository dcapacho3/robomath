"""Robomath - The Lie group Sim(3) of similarity transforms (rotate, scale, translate).

A similarity moves a point as p -> s R p + t, with s > 0. It is the pose group of a
camera that does not know the true size of the world (monocular SLAM), and the
group a point-set fit with a scale lives in (see umeyama).

A pose is the 4x4 matrix M = [[s R, t], [0, 1]]. A tangent vector is
tau = (rho, theta, sigma) in R^7: the SE(3) part first, then the log of the scale,
so s = exp(sigma). Hat gives [[sigma I + [theta]x, rho], [0, 0]], and
Exp(tau) = [[exp(sigma) Exp(theta), W rho], [0, 1]] with
W = sum over n of A^n / (n + 1)!, A = sigma I + [theta]x. W is the SO(3) matrix V of
SE(3) when sigma = 0. The formulas follow Strasdat's thesis (2012) and Sophus; the
tests check them against scipy's matrix exponential and logarithm.

Every function accepts leading batch dimensions.
"""

import numpy as np

from ._util import apply
from .se3 import angular, linear
from .so3 import exp_vec, hat, log_vec, vee

SCALE = 6  # position of sigma in a tangent vector (after the six SE(3) entries)
SQUARINGS = 6  # W is computed for A / 2^SQUARINGS, then doubled back
TERMS = 12  # Taylor terms for the scaled-down matrix


def _phi1(a: np.ndarray) -> np.ndarray:
    """W = sum of A^n / (n + 1)! for 3x3 matrices, shape (..., 3, 3).

    Taylor series for A / 2^k, then k doublings with phi1(2B) = phi1(B)(I + e^B)/2.
    No special case at zero, and accurate for the size of A that poses give.
    """
    eye = np.broadcast_to(np.eye(3), a.shape)
    b = a / 2**SQUARINGS
    term, exp_b, phi = eye, eye, eye
    for n in range(1, TERMS):
        term = term @ b / n  # B^n / n!
        exp_b = exp_b + term
        phi = phi + term / (n + 1)
    for _ in range(SQUARINGS):
        phi = phi @ (eye + exp_b) / 2
        exp_b = exp_b @ exp_b
    return phi


def srt_to_sim3(s: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 similarity [[s R, t], [0, 1]] from a scale, rotation and translation.

    Args:
        s (np.ndarray): Scales, shape (...), positive.
        r (np.ndarray): Rotation matrices, shape (..., 3, 3).
        t (np.ndarray): Translations, shape (..., 3).

    Returns:
        np.ndarray: The poses, shape (..., 4, 4).
    """
    s, r, t = (np.asarray(x, dtype=float) for x in (s, r, t))
    m = np.zeros(np.broadcast_shapes(s.shape, r.shape[:-2], t.shape[:-1]) + (4, 4))
    m[..., :3, :3] = s[..., None, None] * r
    m[..., :3, 3] = t
    m[..., 3, 3] = 1.0
    return m


def sim3_to_srt(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale, rotation matrix and translation of a 4x4 similarity.

    Args:
        m (np.ndarray): Poses, shape (..., 4, 4).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (s, R, t), shapes (...),
            (..., 3, 3) and (..., 3).
    """
    m = np.asarray(m, dtype=float)
    a = m[..., :3, :3]
    s = np.cbrt(np.linalg.det(a))  # det(s R) = s^3
    return s, a / s[..., None, None], m[..., :3, 3].copy()


def sim3_inv(m: np.ndarray) -> np.ndarray:
    """Inverse of a similarity: [[R^T / s, -R^T t / s], [0, 1]].

    Args:
        m (np.ndarray): Poses, shape (..., 4, 4).

    Returns:
        np.ndarray: The inverse poses, shape (..., 4, 4).
    """
    m = np.asarray(m, dtype=float)
    a, t = m[..., :3, :3], m[..., :3, 3]
    s2 = np.trace(a @ np.swapaxes(a, -1, -2), axis1=-2, axis2=-1) / 3  # s^2
    a_inv = np.swapaxes(a, -1, -2) / s2[..., None, None]  # (s R)^-1 = (s R)^T / s^2
    out = np.zeros(m.shape)
    out[..., :3, :3] = a_inv
    out[..., :3, 3] = -apply(a_inv, t)
    out[..., 3, 3] = 1.0
    return out


def act_sim3(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Move points: p_a = s R p_b + t. Shapes (..., 4, 4), (..., N, 3)."""
    a, t = m[..., :3, :3], m[..., :3, 3]
    return points @ np.swapaxes(a, -1, -2) + t[..., None, :]


def hat_sim3(tau: np.ndarray) -> np.ndarray:
    """4x4 matrix [[sigma I + [theta]x, rho], [0, 0]] of a tangent vector (..., 7)."""
    tau = np.asarray(tau, dtype=float)
    m = np.zeros(tau.shape[:-1] + (4, 4))
    m[..., :3, :3] = hat(angular(tau)) + tau[..., SCALE, None, None] * np.eye(3)
    m[..., :3, 3] = linear(tau)
    return m


def vee_sim3(m: np.ndarray) -> np.ndarray:
    """Tangent vector (rho, theta, sigma) of a 4x4 sim(3) matrix. Inverts hat_sim3."""
    m = np.asarray(m, dtype=float)
    sigma = np.trace(m[..., :3, :3], axis1=-2, axis2=-1) / 3
    return np.concatenate(
        [m[..., :3, 3], vee(m[..., :3, :3]), sigma[..., None]], axis=-1
    )


def exp_vec_sim3(tau: np.ndarray) -> np.ndarray:
    """Similarity of a tangent vector (..., 7): [[e^sigma Exp(theta), W rho], ...]."""
    tau = np.asarray(tau, dtype=float)
    theta, sigma = angular(tau), tau[..., SCALE]
    a = hat(theta) + sigma[..., None, None] * np.eye(3)
    return srt_to_sim3(np.exp(sigma), exp_vec(theta), apply(_phi1(a), linear(tau)))


def log_vec_sim3(m: np.ndarray) -> np.ndarray:
    """Tangent vector (rho, theta, sigma) of a similarity, with |theta| <= pi."""
    s, r, t = sim3_to_srt(m)
    theta, sigma = log_vec(r), np.log(s)
    w = _phi1(hat(theta) + sigma[..., None, None] * np.eye(3))
    rho = np.linalg.solve(w, t[..., None])[..., 0]
    return np.concatenate([rho, theta, sigma[..., None]], axis=-1)


def adjoint_sim3(m: np.ndarray) -> np.ndarray:
    """7x7 adjoint [[s R, [t]x R, -t], [0, R, 0], [0, 0, 1]] for (rho, theta, sigma)."""
    s, r, t = sim3_to_srt(m)
    out = np.zeros(r.shape[:-2] + (7, 7))
    out[..., :3, :3] = s[..., None, None] * r
    out[..., :3, 3:6] = hat(t) @ r
    out[..., :3, SCALE] = -t
    out[..., 3:6, 3:6] = r
    out[..., SCALE, SCALE] = 1.0
    return out


def rplus_sim3(m: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Right plus M (+) tau: move M by tau in its own frame."""
    return m @ exp_vec_sim3(tau)


def rminus_sim3(m2: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Right minus M2 (-) M1: the local tangent vector from M1 to M2."""
    return log_vec_sim3(sim3_inv(m1) @ m2)


def lplus_sim3(tau: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Left plus tau (+) M: move M by tau in the global frame."""
    return exp_vec_sim3(tau) @ m


def lminus_sim3(m2: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Left minus M2 (-) M1: the global tangent vector from M1 to M2."""
    return log_vec_sim3(m2 @ sim3_inv(m1))

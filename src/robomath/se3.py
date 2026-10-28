"""Robomath - the Lie group SE(3) of rigid motions (Sola et al., Appendix D).

Equation numbers are from arXiv:1812.01537v9. A pose is a 4x4 matrix
M = [[R, t], [0, 1]]. A tangent vector (twist) is tau = (rho, theta) in R^6: the
translation part first, then the rotation vector, as in the paper (eq. 169).

Frame naming: T_ab is the pose of frame b in frame a, so p_a = T_ab @ p_b. In a
product the inner letters cancel, T_ac = T_ab @ T_bc, and the inverse swaps them,
T_ba = trans_inv(T_ab). A variable can be spelled a_from_b for the same matrix.

Shapes: every function accepts leading batch dimensions. A pose is (..., 4, 4), a
rotation (..., 3, 3), a translation (..., 3) and a tangent vector (..., 6). The
batch dimensions broadcast, and a single input is the case with none.

Layout: only this module knows which half of a tangent vector is which. Other code
builds a twist with twist() and reads it with linear() and angular(), so the order
can be read in one place (LINEAR and ANGULAR below).
"""

import numpy as np

from ._util import apply
from .so3 import exp_vec, hat, jl, jl_inv, log_vec, vee

# The one place that fixes the order of a tangent vector (eq. 169).
LINEAR = slice(0, 3)
ANGULAR = slice(3, 6)


def twist(rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Tangent vector from its translation part and its rotation vector.

    Args:
        rho (np.ndarray): Translation part, shape (..., 3).
        theta (np.ndarray): Rotation vector, shape (..., 3).

    Returns:
        np.ndarray: The tangent vector, shape (..., 6).
    """
    return np.concatenate([rho, theta], axis=-1)


def linear(xi: np.ndarray) -> np.ndarray:
    """Translation part rho of a tangent vector, shape (..., 3)."""
    return xi[..., LINEAR]


def angular(xi: np.ndarray) -> np.ndarray:
    """Rotation vector theta of a tangent vector, shape (..., 3)."""
    return xi[..., ANGULAR]


def rp_to_trans(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pose from a rotation matrix and a translation (eq. 168). Also works for SE(2).

    Args:
        r (np.ndarray): Rotation matrices, shape (..., 3, 3).
        t (np.ndarray): Translations, shape (..., 3).

    Returns:
        np.ndarray: The poses [[R, t], [0, 1]], shape (..., 4, 4).
    """
    r, t = np.asarray(r, dtype=float), np.asarray(t, dtype=float)
    n = r.shape[-1]  # 3 for SE(3); the same code serves SE(2) with n = 2
    m = np.zeros(np.broadcast_shapes(r.shape[:-2], t.shape[:-1]) + (n + 1, n + 1))
    m[..., :n, :n] = r
    m[..., :n, n] = t
    m[..., n, n] = 1.0
    return m


def trans_to_rp(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation matrix and translation of a pose. Also works for SE(2).

    Args:
        m (np.ndarray): Poses, shape (..., 4, 4).

    Returns:
        tuple[np.ndarray, np.ndarray]: (R, t), shapes (..., 3, 3) and (..., 3),
            copies of the blocks of m.
    """
    m = np.asarray(m)
    n = m.shape[-1] - 1
    return m[..., :n, :n].copy(), m[..., :n, n].copy()


def trans_inv(m: np.ndarray) -> np.ndarray:
    """Inverse of a pose (eq. 170): [[R^T, -R^T t], [0, 1]]. Also works for SE(2).

    Args:
        m (np.ndarray): Poses, shape (..., 4, 4).

    Returns:
        np.ndarray: The inverse poses, shape (..., 4, 4).
    """
    r, t = trans_to_rp(m)
    rt = np.swapaxes(r, -1, -2)
    return rp_to_trans(rt, -apply(rt, t))


def act_se3(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Move points by poses: p_a = R p_b + t. Shapes (..., 4, 4), (..., N, 3)."""
    r, t = trans_to_rp(m)
    return points @ np.swapaxes(r, -1, -2) + t[..., None, :]


def hat_se3(tau: np.ndarray) -> np.ndarray:
    """4x4 matrix of a tangent vector tau = (rho, theta) (eq. 169).

    Args:
        tau (np.ndarray): Tangent vectors, shape (..., 6).

    Returns:
        np.ndarray: [[hat(theta), rho], [0, 0]], shape (..., 4, 4).
    """
    tau = np.asarray(tau, dtype=float)
    m = np.zeros(tau.shape[:-1] + (4, 4))
    m[..., :3, :3] = hat(angular(tau))
    m[..., :3, 3] = linear(tau)
    return m


def vee_se3(m: np.ndarray) -> np.ndarray:
    """Tangent vector (rho, theta) of a 4x4 se(3) matrix. Inverse of hat_se3.

    Args:
        m (np.ndarray): Matrices [[hat(theta), rho], [0, 0]], shape (..., 4, 4).

    Returns:
        np.ndarray: Vectors, shape (..., 6).
    """
    m = np.asarray(m, dtype=float)
    return twist(m[..., :3, 3], vee(m[..., :3, :3]))


def exp_vec_se3(tau: np.ndarray) -> np.ndarray:
    """Pose of a tangent vector: the paper's Exp for SE(3) (eq. 172, 174).

    Args:
        tau (np.ndarray): Tangent vectors, shape (..., 6).

    Returns:
        np.ndarray: The poses [[Exp(theta), V(theta) rho], [0, 1]], (..., 4, 4).
    """
    tau = np.asarray(tau, dtype=float)
    theta = angular(tau)
    # V(theta) of eq. 174 is the left Jacobian of SO(3).
    return rp_to_trans(exp_vec(theta), apply(jl(theta), linear(tau)))


def log_vec_se3(m: np.ndarray) -> np.ndarray:
    """Tangent vector of a pose: the paper's Log for SE(3) (eq. 173).

    Args:
        m (np.ndarray): Poses, shape (..., 4, 4).

    Returns:
        np.ndarray: Tangent vectors, shape (..., 6), with |theta| <= pi.
    """
    r, t = trans_to_rp(m)
    theta = log_vec(r)
    return twist(apply(jl_inv(theta), t), theta)


def adjoint(m: np.ndarray) -> np.ndarray:
    """6x6 adjoint matrix of a pose (eq. 175), for tangent vectors (rho, theta).

    Maps a tangent vector in the local frame to the same motion in the global
    frame: exp_vec_se3(adjoint(M) @ tau) == M @ exp_vec_se3(tau) @ trans_inv(M).

    Args:
        m (np.ndarray): Poses, shape (..., 4, 4).

    Returns:
        np.ndarray: The matrices [[R, [t]x R], [0, R]], shape (..., 6, 6).
    """
    r, t = trans_to_rp(m)
    out = np.zeros(r.shape[:-2] + (6, 6))
    out[..., LINEAR, LINEAR] = r
    out[..., LINEAR, ANGULAR] = hat(t) @ r
    out[..., ANGULAR, ANGULAR] = r
    return out


def rplus_se3(m: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Right plus M (+) tau: move pose M by tau in its own frame (eq. 25)."""
    return m @ exp_vec_se3(tau)


def rminus_se3(m2: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Right minus M2 (-) M1: the local tangent vector from M1 to M2 (eq. 26)."""
    return log_vec_se3(trans_inv(m1) @ m2)


def lplus_se3(tau: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Left plus tau (+) M: move pose M by tau in the global frame (eq. 27)."""
    return exp_vec_se3(tau) @ m


def lminus_se3(m2: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Left minus M2 (-) M1: the global tangent vector from M1 to M2 (eq. 28)."""
    return log_vec_se3(m2 @ trans_inv(m1))

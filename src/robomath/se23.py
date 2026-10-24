"""Robomath - The Lie group SE_2(3) of extended poses (orientation, velocity, position).

The state of a vehicle with an IMU: a rotation R, a velocity v and a position p, in
the 5x5 matrix M = [[R, v, p], [0, 1, 0], [0, 0, 1]]. Velocity and position rotate
together, which is why the three form one group; the invariant EKF and IMU
preintegration use it (Barrau and Bonnabel 2018; Brossard et al., arXiv:2007.14097).

A tangent vector is tau = (rho, theta, nu) in R^9: position part, rotation vector,
velocity part. The first six entries are the SE(3) tangent vector, with the extra
velocity appended. Exp(tau) = [[Exp(theta), V nu, V rho], ...] with V the left
Jacobian of SO(3), the same V as in SE(3).

Every function accepts leading batch dimensions.
"""

import numpy as np

from ._util import apply
from .se3 import angular, linear
from .so3 import exp_vec, hat, jl, jl_inv, log_vec, vee

VELOCITY = slice(6, 9)  # the velocity part of a tangent vector


def rvp_to_se23(r: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """5x5 extended pose from a rotation matrix, a velocity and a position.

    Args:
        r (np.ndarray): Rotation matrices, shape (..., 3, 3).
        v (np.ndarray): Velocities, shape (..., 3).
        p (np.ndarray): Positions, shape (..., 3).

    Returns:
        np.ndarray: The poses [[R, v, p], [0, 1, 0], [0, 0, 1]], shape (..., 5, 5).
    """
    r, v, p = (np.asarray(x, dtype=float) for x in (r, v, p))
    batch = np.broadcast_shapes(r.shape[:-2], v.shape[:-1], p.shape[:-1])
    m = np.zeros(batch + (5, 5))
    m[..., :3, :3] = r
    m[..., :3, 3] = v
    m[..., :3, 4] = p
    m[..., 3, 3] = 1.0
    m[..., 4, 4] = 1.0
    return m


def se23_to_rvp(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation matrix, velocity and position of a 5x5 extended pose (copies).

    Args:
        m (np.ndarray): Poses, shape (..., 5, 5).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (R, v, p), shapes (..., 3, 3),
            (..., 3) and (..., 3).
    """
    m = np.asarray(m)
    return m[..., :3, :3].copy(), m[..., :3, 3].copy(), m[..., :3, 4].copy()


def se23_inv(m: np.ndarray) -> np.ndarray:
    """Inverse of an extended pose: [[R^T, -R^T v, -R^T p], ...].

    Args:
        m (np.ndarray): Poses, shape (..., 5, 5).

    Returns:
        np.ndarray: The inverse poses, shape (..., 5, 5).
    """
    r, v, p = se23_to_rvp(m)
    rt = np.swapaxes(r, -1, -2)
    return rvp_to_se23(rt, -apply(rt, v), -apply(rt, p))


def hat_se23(tau: np.ndarray) -> np.ndarray:
    """5x5 matrix [[[theta]x, nu, rho], [0, 0, 0], [0, 0, 0]] of a tangent vector."""
    tau = np.asarray(tau, dtype=float)
    m = np.zeros(tau.shape[:-1] + (5, 5))
    m[..., :3, :3] = hat(angular(tau))
    m[..., :3, 3] = tau[..., VELOCITY]
    m[..., :3, 4] = linear(tau)
    return m


def vee_se23(m: np.ndarray) -> np.ndarray:
    """Tangent vector (rho, theta, nu) of a 5x5 se_2(3) matrix. Inverse of hat_se23."""
    m = np.asarray(m, dtype=float)
    return np.concatenate([m[..., :3, 4], vee(m[..., :3, :3]), m[..., :3, 3]], axis=-1)


def exp_vec_se23(tau: np.ndarray) -> np.ndarray:
    """Extended pose of a tangent vector (..., 9): rotation Exp(theta), V nu, V rho."""
    tau = np.asarray(tau, dtype=float)
    theta = angular(tau)
    v = jl(theta)
    return rvp_to_se23(
        exp_vec(theta), apply(v, tau[..., VELOCITY]), apply(v, linear(tau))
    )


def log_vec_se23(m: np.ndarray) -> np.ndarray:
    """Tangent vector (rho, theta, nu) of an extended pose, with |theta| <= pi."""
    r, v, p = se23_to_rvp(m)
    theta = log_vec(r)
    v_inv = jl_inv(theta)
    return np.concatenate([apply(v_inv, p), theta, apply(v_inv, v)], axis=-1)


def adjoint_se23(m: np.ndarray) -> np.ndarray:
    """9x9 adjoint [[R, [p]x R, 0], [0, R, 0], [0, [v]x R, R]] for (rho, theta, nu)."""
    r, v, p = se23_to_rvp(m)
    out = np.zeros(r.shape[:-2] + (9, 9))
    out[..., :3, :3] = r
    out[..., :3, 3:6] = hat(p) @ r
    out[..., 3:6, 3:6] = r
    out[..., 6:9, 3:6] = hat(v) @ r
    out[..., 6:9, 6:9] = r
    return out


def rplus_se23(m: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Right plus M (+) tau: move M by tau in its own frame."""
    return m @ exp_vec_se23(tau)


def rminus_se23(m2: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Right minus M2 (-) M1: the local tangent vector from M1 to M2."""
    return log_vec_se23(se23_inv(m1) @ m2)


def lplus_se23(tau: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Left plus tau (+) M: move M by tau in the global frame."""
    return exp_vec_se23(tau) @ m


def lminus_se23(m2: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Left minus M2 (-) M1: the global tangent vector from M1 to M2."""
    return log_vec_se23(m2 @ se23_inv(m1))

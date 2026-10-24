"""Robomath - Fit a rigid motion to matched points (Kabsch and Umeyama).

Given points p_i and q_i that are the same physical points seen in two frames, find
the motion with q_i ~ R p_i + t that makes the sum of squared distances smallest.
The point sets have shape (..., N, 3), with N >= 3 points not on one line.
Every function accepts leading batch dimensions.
"""

import numpy as np


def _fit(p: np.ndarray, q: np.ndarray):
    """Shared steps: centre p, SVD of the cross-covariance, fix reflections.

    Returns R, the sum of the corrected singular values, the sum of squared lengths
    of the centred p, and the two centres.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p_mean, q_mean = p.mean(axis=-2), q.mean(axis=-2)
    pc = p - p_mean[..., None, :]
    # H = sum of pc_i q_i^T. The centred p sums to zero, so q needs no centring here.
    h = np.swapaxes(pc, -1, -2) @ q
    u, s, vt = np.linalg.svd(h)
    v, ut = np.swapaxes(vt, -1, -2), np.swapaxes(u, -1, -2)
    # D flips the weakest direction when V U^T is a mirror (determinant -1).
    d = np.ones(s.shape)
    d[..., 2] = np.where(np.linalg.det(v @ ut) < 0, -1.0, 1.0)
    r = (v * d[..., None, :]) @ ut
    return r, np.sum(s * d, axis=-1), np.sum(pc * pc, axis=(-2, -1)), p_mean, q_mean


def kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation that best take the points p onto q.

    Args:
        p (np.ndarray): Source points, shape (..., N, 3).
        q (np.ndarray): Matching target points, shape (..., N, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: (R, t) with q ~ R p + t, shapes (..., 3, 3)
            and (..., 3). R is always a rotation, never a mirror.
    """
    r, _, _, p_mean, q_mean = _fit(p, q)
    return r, q_mean - np.einsum("...ij,...j->...i", r, p_mean)


def umeyama(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale, rotation and translation that best take the points p onto q.

    Args:
        p (np.ndarray): Source points, shape (..., N, 3), not all equal.
        q (np.ndarray): Matching target points, shape (..., N, 3).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (s, R, t) with q ~ s R p + t,
            shapes (...), (..., 3, 3) and (..., 3).
    """
    r, trace, spread, p_mean, q_mean = _fit(p, q)
    s = trace / spread
    return s, r, q_mean - s[..., None] * np.einsum("...ij,...j->...i", r, p_mean)

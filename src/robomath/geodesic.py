"""Robomath - Distance, interpolation and mean on SO(3) and SE(3).

All of them follow the shortest path (the geodesic) given by the right plus and
minus of the group: the step from A to B is minus(B, A), and moving a fraction t
of it is plus(A, t * minus(B, A)). Every function accepts leading batch dimensions.
"""

import warnings

import numpy as np

from ._util import norm
from .quaternions import matrix_to_quat, quat_to_matrix
from .se2 import rminus_se2, rplus_se2
from .se3 import angular, rminus_se3, rp_to_trans, rplus_se3, trans_to_rp
from .so2 import exp_so2, log_so2
from .so3 import rminus, rplus

MEAN_TOL = 1e-13
MEAN_ITERATIONS = 100
UNIQUE_ANGLE = np.pi / 2


def distance_so3(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Angle in [0, pi] of the rotation that takes r1 to r2, shape (...)."""
    return norm(rminus(r2, r1))


def interp_so3(r1: np.ndarray, r2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rotation a fraction t of the way from r1 (t = 0) to r2 (t = 1)."""
    t = np.asarray(t, dtype=float)
    return rplus(r1, t[..., None] * rminus(r2, r1))


def interp_se2(m1: np.ndarray, m2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Planar pose a fraction t of the way from m1 (t = 0) to m2 (t = 1)."""
    t = np.asarray(t, dtype=float)
    return rplus_se2(m1, t[..., None] * rminus_se2(m2, m1))


def interp_se3(m1: np.ndarray, m2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pose a fraction t of the way from m1 (t = 0) to m2 (t = 1)."""
    t = np.asarray(t, dtype=float)
    return rplus_se3(m1, t[..., None] * rminus_se3(m2, m1))


def _weighted(w, x):
    """Weighted sum over the N items: w (..., N) times x (..., N, k) -> (..., k)."""
    return np.einsum("...n,...nk->...k", w, x)


def _start_so3(rotations, w):
    """Chordal mean: the top eigenvector of the weighted sum of q q^T (Markley).

    No iteration and no local minima, and it does not depend on the item order.
    Same mean as scipy's Rotation.mean, written here from the method.
    """
    q = matrix_to_quat(rotations)
    k = np.einsum("...n,...ni,...nj->...ij", w, q, q)
    return quat_to_matrix(np.linalg.eigh(k)[1][..., -1])


def _start_se3(poses, w):
    r, t = trans_to_rp(poses)
    return rp_to_trans(_start_so3(r, w), _weighted(w, t))


def _start_se2(poses, w):
    r, t = trans_to_rp(poses)
    theta = log_so2(r)
    mean_angle = np.arctan2(
        np.einsum("...n,...n->...", w, np.sin(theta)),
        np.einsum("...n,...n->...", w, np.cos(theta)),
    )
    return rp_to_trans(exp_so2(mean_angle), _weighted(w, t))


def _angle_se3(tau):
    """Rotation angle of each SE(3) tangent vector."""
    return norm(angular(tau))


def _angle_se2(tau):
    """Rotation angle of each SE(2) tangent vector."""
    return np.abs(tau[..., 2])


def _mean(items, weights, start, plus, minus, angle):
    """Karcher mean: move the guess by the weighted mean of its offsets to the items.

    items has shape (..., N, a, a); the N items are averaged. The first guess is
    the chordal mean of the rotations, so the answer does not depend on the item
    order. angle maps tangent offsets to the rotation angle of each. The mean is
    unique when all items lie within 90 degrees of it. Beyond that the loop can
    settle on a local answer, so it warns.
    """
    items = np.asarray(items, dtype=float)
    n = items.shape[-3]
    if n == 0:
        raise ValueError("the mean of no items is undefined")
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    if np.any(w < 0) or np.any(np.sum(w, axis=-1) <= 0):
        raise ValueError("weights must be non-negative and not all zero")
    w = w / np.sum(w, axis=-1, keepdims=True)
    mean = start(items, w)
    for _ in range(MEAN_ITERATIONS):
        step = np.sum(w[..., None] * minus(items, mean[..., None, :, :]), axis=-2)
        mean = plus(mean, step)
        if np.max(np.abs(step), initial=0.0) < MEAN_TOL:
            offsets = minus(items, mean[..., None, :, :])
            if np.max(angle(offsets), initial=0.0) >= UNIQUE_ANGLE:
                warnings.warn(
                    "an item is 90 degrees or more from the mean; "
                    "the mean may not be unique",
                    stacklevel=3,
                )
            return mean
    raise ValueError("mean did not settle; the items are too spread out")


def mean_so3(rotations: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Mean of N rotations, shape (..., N, 3, 3) -> (..., 3, 3).

    Args:
        rotations (np.ndarray): The rotations to average.
        weights (np.ndarray | None): Weights, shape (..., N). Equal if None.
    """
    return _mean(rotations, weights, _start_so3, rplus, rminus, norm)


def mean_se3(poses: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Mean of N poses, shape (..., N, 4, 4) -> (..., 4, 4).

    Args:
        poses (np.ndarray): The poses to average.
        weights (np.ndarray | None): Weights, shape (..., N). Equal if None.
    """
    return _mean(poses, weights, _start_se3, rplus_se3, rminus_se3, _angle_se3)


def mean_se2(poses: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Mean of N planar poses, shape (..., N, 3, 3) -> (..., 3, 3).

    Args:
        poses (np.ndarray): The poses to average.
        weights (np.ndarray | None): Weights, shape (..., N). Equal if None.
    """
    return _mean(poses, weights, _start_se2, rplus_se2, rminus_se2, _angle_se2)

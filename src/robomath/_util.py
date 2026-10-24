"""Robomath - small helpers shared by the other modules."""

import numpy as np


def norm(v: np.ndarray) -> np.ndarray:
    """Length of the vectors along the last axis, shape (...).

    Same result as np.linalg.norm(v, axis=-1)
    """
    return np.sqrt(np.einsum("...i,...i->...", v, v))


def apply(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Matrix-vector product for stacks: a (..., n, n) times v (..., n)."""
    return (a @ v[..., None])[..., 0]

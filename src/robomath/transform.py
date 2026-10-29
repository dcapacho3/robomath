"""Robomath - The Transform class: an SE(3) pose with operators.

A thin wrapper over the functions of se3.py. It holds a 4x4 matrix (or a batch of
them, shape (..., 4, 4)) and follows the frame naming of se3.py: a variable named
t_ab is the pose of frame b in frame a, so t_ac = t_ab @ t_bc and
t_ba = t_ab.inverse().
"""

from dataclasses import dataclass

import numpy as np

from .convert import pq_to_trans
from .geodesic import interp_se3
from .se3 import (
    adjoint,
    exp_vec_se3,
    lminus_se3,
    log_vec_se3,
    lplus_se3,
    rminus_se3,
    rp_to_trans,
    rplus_se3,
    trans_inv,
    trans_to_rp,
)
from .so3 import nearest_rotation


@dataclass(frozen=True, eq=False)
class Transform:
    """A rigid motion of 3D space: p_a = T_ab @ p_b.

    Attributes:
        matrix (np.ndarray): The 4x4 pose, or a stack (..., 4, 4). Stored as a
            read-only float copy. The constructor trusts it to be a valid pose; use
            from_matrix for data from outside, which repairs a drifted rotation.
    """

    matrix: np.ndarray

    # Make numpy hand `array @ transform` back to us instead of looping over elements.
    __array_ufunc__ = None

    def __post_init__(self):
        """Check the shape and store a read-only float copy of the matrix."""
        m = np.array(self.matrix, dtype=float)
        if m.ndim < 2 or m.shape[-2:] != (4, 4):
            raise ValueError(f"expected shape (..., 4, 4), got {m.shape}")
        m.flags.writeable = False
        object.__setattr__(self, "matrix", m)

    @classmethod
    def identity(cls) -> "Transform":
        """The pose that moves nothing."""
        return cls(np.eye(4))

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> "Transform":
        """Pose from an unchecked 4x4 matrix: repairs the rotation and bottom row."""
        r, t = trans_to_rp(m)
        return cls(rp_to_trans(nearest_rotation(r), t))

    @classmethod
    def from_rt(cls, r: np.ndarray, t: np.ndarray) -> "Transform":
        """Pose from a rotation matrix (..., 3, 3) and a translation (..., 3)."""
        return cls(rp_to_trans(r, t))

    @classmethod
    def from_pq(cls, p: np.ndarray, q: np.ndarray) -> "Transform":
        """Pose from a position and an (x, y, z, w) quaternion, as in a ROS Pose."""
        return cls(pq_to_trans(p, q))

    @classmethod
    def exp(cls, tau: np.ndarray) -> "Transform":
        """Pose of a tangent vector (translation, rotation), shape (..., 6)."""
        return cls(exp_vec_se3(tau))

    @property
    def rotation(self) -> np.ndarray:
        """Rotation matrix R, shape (..., 3, 3)."""
        return trans_to_rp(self.matrix)[0]

    @property
    def translation(self) -> np.ndarray:
        """Translation t, shape (..., 3)."""
        return trans_to_rp(self.matrix)[1]

    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape: () for one pose."""
        return self.matrix.shape[:-2]

    def inverse(self) -> "Transform":
        """T_ba from T_ab."""
        return Transform(trans_inv(self.matrix))

    def log(self) -> np.ndarray:
        """Tangent vector (translation, rotation) of this pose, shape (..., 6)."""
        return log_vec_se3(self.matrix)

    def adjoint(self) -> np.ndarray:
        """6x6 adjoint matrix, shape (..., 6, 6)."""
        return adjoint(self.matrix)

    def __matmul__(self, other):
        """Compose with a Transform, or move points of shape (..., 3)."""
        if isinstance(other, Transform):
            return Transform(self.matrix @ other.matrix)
        points = np.asarray(other, dtype=float)
        if points.shape[-1] != 3:
            return NotImplemented
        r, t = trans_to_rp(self.matrix)
        return (r @ points[..., None])[..., 0] + t

    def rplus(self, tau: np.ndarray) -> "Transform":
        """Move this pose by tau in its own frame (right plus)."""
        return Transform(rplus_se3(self.matrix, tau))

    def rminus(self, other: "Transform") -> np.ndarray:
        """Tangent vector in this pose's frame that leads from other to this one."""
        return rminus_se3(self.matrix, other.matrix)

    def lplus(self, tau: np.ndarray) -> "Transform":
        """Move this pose by tau in the global frame (left plus)."""
        return Transform(lplus_se3(tau, self.matrix))

    def lminus(self, other: "Transform") -> np.ndarray:
        """Tangent vector in the global frame that leads from other to this pose."""
        return lminus_se3(self.matrix, other.matrix)

    def interp(self, other: "Transform", t: np.ndarray) -> "Transform":
        """Pose a fraction t of the way from this one (t = 0) to other (t = 1)."""
        return Transform(interp_se3(self.matrix, other.matrix, t))

    def isclose(self, other: "Transform", atol: float = 1e-9) -> bool:
        """True if every matrix entry is within atol of the other's."""
        return bool(np.allclose(self.matrix, other.matrix, rtol=0, atol=atol))

    def __eq__(self, other) -> bool:
        """Exact equality of the matrices."""
        return isinstance(other, Transform) and np.array_equal(
            self.matrix, other.matrix
        )

    __hash__ = None

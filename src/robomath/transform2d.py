"""Robomath - The Transform2D class: an SE(2) pose with operators.

The planar twin of Transform. It holds a 3x3 matrix (or a batch of them, shape
(..., 3, 3)) and follows the same frame naming: a variable named t_ab is the pose
of frame b in frame a, so t_ac = t_ab @ t_bc and t_ba = t_ab.inverse().
"""

from dataclasses import dataclass

import numpy as np

from .se2 import (
    adjoint_se2,
    exp_vec_se2,
    lminus_se2,
    log_vec_se2,
    lplus_se2,
    rminus_se2,
    rplus_se2,
    se2_to_xyt,
    xyt_to_se2,
)
from .se3 import rp_to_trans, trans_inv, trans_to_rp
from .so2 import exp_so2, log_so2


@dataclass(frozen=True, eq=False)
class Transform2D:
    """A rigid motion of the plane: p_a = T_ab @ p_b.

    Attributes:
        matrix (np.ndarray): The 3x3 pose, or a stack (..., 3, 3). Stored as a
            read-only float copy. The constructor trusts it to be a valid pose; use
            from_matrix for data from outside, which repairs a drifted rotation.
    """

    matrix: np.ndarray

    # Make numpy hand `array @ transform` back to us instead of looping over elements.
    __array_ufunc__ = None

    def __post_init__(self):
        """Check the shape and store a read-only float copy of the matrix."""
        m = np.array(self.matrix, dtype=float)
        if m.ndim < 2 or m.shape[-2:] != (3, 3):
            raise ValueError(f"expected shape (..., 3, 3), got {m.shape}")
        m.flags.writeable = False
        object.__setattr__(self, "matrix", m)

    @classmethod
    def identity(cls) -> "Transform2D":
        """The pose that moves nothing."""
        return cls(np.eye(3))

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> "Transform2D":
        """Pose from an unchecked 3x3 matrix: repairs the rotation and bottom row."""
        r, t = trans_to_rp(m)
        return cls(rp_to_trans(exp_so2(log_so2(r)), t))

    @classmethod
    def from_rt(cls, r: np.ndarray, t: np.ndarray) -> "Transform2D":
        """Pose from a rotation matrix (..., 2, 2) and a translation (..., 2)."""
        return cls(rp_to_trans(r, t))

    @classmethod
    def from_xyt(cls, xyt: np.ndarray) -> "Transform2D":
        """Pose from (x, y, angle), shape (..., 3), as odometry reports it."""
        return cls(xyt_to_se2(xyt))

    @classmethod
    def exp(cls, tau: np.ndarray) -> "Transform2D":
        """Pose of a tangent vector (x, y, angle), shape (..., 3)."""
        return cls(exp_vec_se2(tau))

    @property
    def rotation(self) -> np.ndarray:
        """Rotation matrix R, shape (..., 2, 2)."""
        return trans_to_rp(self.matrix)[0]

    @property
    def translation(self) -> np.ndarray:
        """Translation t, shape (..., 2)."""
        return trans_to_rp(self.matrix)[1]

    @property
    def xyt(self) -> np.ndarray:
        """Coordinates (x, y, angle) with the angle in (-pi, pi], shape (..., 3)."""
        return se2_to_xyt(self.matrix)

    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape: () for one pose."""
        return self.matrix.shape[:-2]

    def inverse(self) -> "Transform2D":
        """T_ba from T_ab."""
        return Transform2D(trans_inv(self.matrix))

    def log(self) -> np.ndarray:
        """Tangent vector (x, y, angle) of this pose, shape (..., 3)."""
        return log_vec_se2(self.matrix)

    def adjoint(self) -> np.ndarray:
        """3x3 adjoint matrix, shape (..., 3, 3)."""
        return adjoint_se2(self.matrix)

    def __matmul__(self, other):
        """Compose with a Transform2D, or move points of shape (..., 2)."""
        if isinstance(other, Transform2D):
            return Transform2D(self.matrix @ other.matrix)
        points = np.asarray(other, dtype=float)
        if points.shape[-1] != 2:
            return NotImplemented
        r, t = trans_to_rp(self.matrix)
        return (r @ points[..., None])[..., 0] + t

    def rplus(self, tau: np.ndarray) -> "Transform2D":
        """Move this pose by tau in its own frame (right plus)."""
        return Transform2D(rplus_se2(self.matrix, tau))

    def rminus(self, other: "Transform2D") -> np.ndarray:
        """Tangent vector in this pose's frame that leads from other to this one."""
        return rminus_se2(self.matrix, other.matrix)

    def lplus(self, tau: np.ndarray) -> "Transform2D":
        """Move this pose by tau in the global frame (left plus)."""
        return Transform2D(lplus_se2(tau, self.matrix))

    def lminus(self, other: "Transform2D") -> np.ndarray:
        """Tangent vector in the global frame that leads from other to this pose."""
        return lminus_se2(self.matrix, other.matrix)

    def isclose(self, other: "Transform2D", atol: float = 1e-9) -> bool:
        """True if every matrix entry is within atol of the other's."""
        return bool(np.allclose(self.matrix, other.matrix, rtol=0, atol=atol))

    def __eq__(self, other) -> bool:
        """Exact equality of the matrices."""
        return isinstance(other, Transform2D) and np.array_equal(
            self.matrix, other.matrix
        )

    __hash__ = None

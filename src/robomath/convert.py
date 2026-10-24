"""Robomath - Convert between robomath's layouts and those of other libraries.

Inside robomath a quaternion is (x, y, z, w) and a tangent vector is (translation,
rotation). Other libraries differ, and a mix-up still runs but gives wrong results.
Each function here changes only the layout, named by the layout and not the library.
Every function accepts leading batch dimensions.

Quaternion (x, y, z, w), scalar last: scipy, ROS messages, Eigen coeffs(), Pinocchio.
Quaternion (w, x, y, z), scalar first: Drake, MuJoCo, Ceres.
Tangent vector (translation, rotation): Sola's paper, Sophus, manif, Pinocchio.
Tangent vector (rotation, translation): Modern Robotics, Drake, MuJoCo, GTSAM Pose3.

Not here yet, to add when a use needs them: poses with the camera convention
(OpenCV and COLMAP store world-to-camera, so the inverse of ours), other Euler
sequences, SE(2) layouts such as GTSAM's, and ROS message classes (this module takes
plain arrays, so it needs no ROS install).
"""

import numpy as np

from .quaternions import matrix_to_quat, quat_to_matrix
from .se3 import angular, linear, rp_to_trans, trans_to_rp, twist


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Move the scalar part from last to first: (x, y, z, w) -> (w, x, y, z)."""
    return np.roll(np.asarray(q, dtype=float), 1, axis=-1)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Move the scalar part from first to last: (w, x, y, z) -> (x, y, z, w)."""
    return np.roll(np.asarray(q, dtype=float), -1, axis=-1)


def swap_twist(tau: np.ndarray) -> np.ndarray:
    """Swap the halves of a tangent vector, (a, b) -> (b, a).

    Turns (translation, rotation) into (rotation, translation) and back; the swap is
    its own inverse. Works for a tangent vector or a spatial velocity.
    """
    tau = np.asarray(tau, dtype=float)
    return twist(angular(tau), linear(tau))


def pq_to_trans(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """4x4 pose from a position and an (x, y, z, w) quaternion, as in a ROS Pose.

    Args:
        p (np.ndarray): Positions, shape (..., 3).
        q (np.ndarray): Unit quaternions, shape (..., 4).

    Returns:
        np.ndarray: The poses, shape (..., 4, 4).
    """
    return rp_to_trans(quat_to_matrix(q), p)


def trans_to_pq(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position and (x, y, z, w) quaternion of a 4x4 pose, as in a ROS Pose.

    Args:
        m (np.ndarray): Poses, shape (..., 4, 4).

    Returns:
        tuple[np.ndarray, np.ndarray]: (p, q), shapes (..., 3) and (..., 4).
    """
    r, t = trans_to_rp(m)
    return t, matrix_to_quat(r)

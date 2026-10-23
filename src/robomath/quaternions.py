"""Robomath - axis-angle and unit quaternions."""
# Quaternion order is xyzw (scalar last)
# Every function accepts leading batch dimensions: quaternions are (..., 4).

import math

import numpy as np

from ._util import norm
from .euler import euler_zyx_to_matrix, matrix_to_euler_zyx
from .so3 import matrix_to_axis_angle

SLERP_TOL = 1 - 1e-12
_CONJUGATE_SIGNS = np.array([-1.0, -1.0, -1.0, 1.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return `q` scaled to unit length (xyzw).

    Boundary helper for noisy sources: floats drift off |q| = 1.

    Args:
        q (np.ndarray): Quaternions, shape (..., 4), none of them zero.

    Returns:
        np.ndarray: q / |q|, shape (..., 4).

    Raises:
        ValueError: If a quaternion is zero (it has no direction).
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        x, y, z, w = q.tolist()
        length = math.sqrt(x * x + y * y + z * z + w * w)
        if length == 0:
            raise ValueError("cannot normalize the zero quaternion")
        return np.array([x / length, y / length, z / length, w / length])
    length = norm(q)[..., None]
    if (length == 0).any():
        raise ValueError("cannot normalize the zero quaternion")
    return q / length


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of `q` (xyzw). For a unit quaternion this is the inverse.

    Args:
        q (np.ndarray): Quaternions, shape (..., 4).

    Returns:
        np.ndarray: (-x, -y, -z, w), shape (..., 4). It rotates by the same angle
            about the reversed axis, which undoes q.
    """
    # axis -> -axis in (axis sin(a/2), cos(a/2)) is the rotation by -angle,
    # because sin is odd and cos is even.
    return np.asarray(q, dtype=float) * _CONJUGATE_SIGNS


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 (xyzw): rotate by q2 first, then by q1.

    Same order as matrices: R(q1 * q2) = R(q1) @ R(q2).

    Args:
        q1 (np.ndarray): Quaternions, shape (..., 4).
        q2 (np.ndarray): Quaternions, shape (..., 4); broadcasts with q1.

    Returns:
        np.ndarray: The products, shape (..., 4). Not commutative.
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    if q1.ndim == 1 and q2.ndim == 1:
        x1, y1, z1, w1 = q1.tolist()
        x2, y2, z2, w2 = q2.tolist()
        return np.array(
            [
                w1 * x2 + w2 * x1 + y1 * z2 - z1 * y2,
                w1 * y2 + w2 * y1 + z1 * x2 - x1 * z2,
                w1 * z2 + w2 * z1 + x1 * y2 - y1 * x2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    p = np.empty(np.broadcast_shapes(q1.shape, q2.shape))
    p[..., 0] = w1 * x2 + w2 * x1 + y1 * z2 - z1 * y2
    p[..., 1] = w1 * y2 + w2 * y1 + z1 * x2 - x1 * z2
    p[..., 2] = w1 * z2 + w2 * z1 + x1 * y2 - y1 * x2
    p[..., 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return p


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit quaternion (xyzw) for a rotation of `angle` radians about unit `axis`.

    Args:
        axis (np.ndarray): Unit vectors, shape (..., 3).
        angle (float): Radians, scalar or shape (...); broadcasts with axis.

    Returns:
        np.ndarray: (axis * sin(angle/2), cos(angle/2)), shape (..., 4).
    """
    axis = np.asarray(axis, dtype=float)
    half = np.asarray(angle, dtype=float) / 2
    if axis.ndim == 1 and half.ndim == 0:
        x, y, z = axis.tolist()
        h = float(half)
        s = math.sin(h)
        return np.array([x * s, y * s, z * s, math.cos(h)])
    q = np.empty(np.broadcast_shapes(axis.shape[:-1], half.shape) + (4,))
    q[..., :3] = axis * np.sin(half)[..., None]
    q[..., 3] = np.cos(half)
    return q


def quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit axis and angle in [0, pi] of a unit quaternion (xyzw).

    Args:
        q (np.ndarray): Unit quaternions, shape (..., 4).

    Returns:
        tuple[np.ndarray, float]: (axis, angle), shapes (..., 3) and (...). At
            angle 0 the axis is undefined and x is returned, like
            matrix_to_axis_angle.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        x, y, z, w = q.tolist()
        if w < 0:
            x, y, z, w = -x, -y, -z, -w
        sin_half = math.sqrt(x * x + y * y + z * z)
        angle = 2 * math.atan2(sin_half, w)
        if sin_half == 0:
            return np.array([1.0, 0.0, 0.0]), angle
        return np.array([x / sin_half, y / sin_half, z / sin_half]), angle
    q = np.where(q[..., 3:] < 0, -q, q)
    sin_half = norm(q[..., :3])
    cos_half = q[..., 3]
    # arctan2, not 2*arccos(cos_half): arccos is flat near cos = 1, so a tiny
    # angle would lose its digits. sin_half is exact there.
    angle = 2 * np.arctan2(sin_half, cos_half)
    axis = q[..., :3] / np.where(sin_half > 0, sin_half, 1.0)[..., None]
    no_rotation = sin_half == 0
    if no_rotation.any():
        axis[no_rotation] = [1.0, 0.0, 0.0]
    return axis, angle


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion (xyzw).

    Args:
        q (np.ndarray): Unit quaternions, shape (..., 4). Not checked (boundary
            rule).

    Returns:
        np.ndarray: The rotation matrices, shape (..., 3, 3).
    """
    # Rodrigues is R = I + sin(a)[w] + (1 - cos(a))[w]^2. With v = w sin(a/2)
    # and s = cos(a/2):
    #   sin(a)     = 2 sin(a/2) cos(a/2)  so  sin(a)[w]        = 2 s [v]
    #   1 - cos(a) = 2 sin^2(a/2)         so  (1 - cos(a))[w]^2 = 2 [v]^2
    # which gives R = I + 2 s [v] + 2 [v]^2, written out entry by entry below.
    # Each entry is a product of two components of q: the 2[v]^2
    # term gives the diagonal 1 - 2(y^2 + z^2) etc. and the symmetric
    # off-diagonal part, 2 s [v] the antisymmetric part.
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        x, y, z, w = q.tolist()
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ]
        )
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = np.empty(x.shape + (3, 3))
    r[..., 0, 0] = 1 - 2 * (yy + zz)
    r[..., 0, 1] = 2 * (xy - wz)
    r[..., 0, 2] = 2 * (xz + wy)
    r[..., 1, 0] = 2 * (xy + wz)
    r[..., 1, 1] = 1 - 2 * (xx + zz)
    r[..., 1, 2] = 2 * (yz - wx)
    r[..., 2, 0] = 2 * (xz - wy)
    r[..., 2, 1] = 2 * (yz + wx)
    r[..., 2, 2] = 1 - 2 * (xx + yy)
    return r


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Unit quaternion (xyzw, scalar >= 0) of a rotation matrix.

    Args:
        r (np.ndarray): Rotation matrices, shape (..., 3, 3).

    Returns:
        np.ndarray: Quaternions, shape (..., 4).
    """
    axis, angle = matrix_to_axis_angle(r)
    return axis_angle_to_quat(axis, angle)


def _slerp_single(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Slerp for two quaternions and one fraction, in plain float arithmetic."""
    x0, y0, z0, w0 = q0.tolist()
    x1, y1, z1, w1 = q1.tolist()
    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    if dot < 0:
        x1, y1, z1, w1, dot = -x1, -y1, -z1, -w1, -dot
    if dot > SLERP_TOL:
        a, b = 1 - t, t
        scale = 1 / math.sqrt(
            sum(
                (a * u + b * v) ** 2
                for u, v in zip((x0, y0, z0, w0), (x1, y1, z1, w1), strict=True)
            )
        )
    else:
        theta = math.acos(min(dot, 1.0))
        sin_theta = math.sin(theta)
        a, b = math.sin((1 - t) * theta) / sin_theta, math.sin(t * theta) / sin_theta
        scale = 1.0
    return np.array(
        [
            scale * (a * x0 + b * x1),
            scale * (a * y0 + b * y1),
            scale * (a * z0 + b * z1),
            scale * (a * w0 + b * w1),
        ]
    )


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation from `q0` (t=0) to `q1` (t=1), xyzw.

    Moves along the shortest arc at constant angular speed. Plain linear
    interpolation of the components (lerp) takes the same path but speeds up
    in the middle, and it is not unit length.

    Args:
        q0 (np.ndarray): Unit quaternions, shape (..., 4).
        q1 (np.ndarray): Unit quaternions, shape (..., 4); broadcasts with q0.
        t (float): 0 to 1, scalar or shape (...); broadcasts with the others.

    Returns:
        np.ndarray: Unit quaternions, shape (..., 4).
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    t = np.asarray(t, dtype=float)
    if q0.ndim == 1 and q1.ndim == 1 and t.ndim == 0:
        return _slerp_single(q0, q1, float(t))
    t = t[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    # q1 and -q1 are the same rotation; take the one on the short side.
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    # Nearly equal: sin(theta) reaches 0 and the sine formula divides by it. A
    # normalized lerp is within ~theta^2 of slerp there (see SLERP_TOL). A batch
    # can mix both cases, chosen by mask.
    near = dot > SLERP_TOL
    theta = np.arccos(np.minimum(dot, 1.0))
    sin_theta = np.where(near, 1.0, np.sin(theta))
    result = (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / sin_theta
    if near.any():
        lerp = (1 - t) * q0 + t * q1
        lerp = lerp / norm(lerp)[..., None]
        result = np.where(near, lerp, result)
    return result


def quat_to_euler_zyx(q: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles (yaw, pitch, roll) of a unit quaternion (xyzw).

    The quaternion has no singularity; the Euler angles do. At pitch = +-pi/2
    yaw is set to 0, as in matrix_to_euler_zyx.

    Args:
        q (np.ndarray): Unit quaternions, shape (..., 4).

    Returns:
        tuple[float, float, float]: (alpha, beta, gamma) = (yaw, pitch, roll)
            in radians, each of shape (...), with R = Rz(alpha) Ry(beta) Rx(gamma).
    """
    return matrix_to_euler_zyx(quat_to_matrix(q))


def euler_zyx_to_quat(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Unit quaternion (xyzw, scalar >= 0) for ZYX Euler angles.

    Args:
        alpha (float): Yaw, angle about z, in radians, scalar or shape (...).
        beta (float): Pitch, angle about y, in radians; broadcasts with alpha.
        gamma (float): Roll, angle about x, in radians; broadcasts with alpha.

    Returns:
        np.ndarray: Quaternions, shape (..., 4), of R = Rz(alpha) Ry(beta) Rx(gamma).
    """
    return matrix_to_quat(euler_zyx_to_matrix(alpha, beta, gamma))

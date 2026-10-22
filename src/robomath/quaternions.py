"""Robomath - axis-angle and unit quaternions."""
# Quaternion order is xyzw (scalar last)

import numpy as np

SIN_TOL = 1e-5
SLERP_TOL = 1 - 1e-12


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for a rotation of `angle` radians about unit `axis`.

    Args:
        axis (np.ndarray): Unit vector, shape (3,).
        angle (float): Radians.

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    wx, wy, wz = axis

    # Construct the skew symmetric matrix
    w = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])

    # Rodrigues' rotation formula
    rot = np.eye(3) + np.sin(angle) * w + (1 - np.cos(angle)) * (w @ w)

    return rot


def matrix_to_axis_angle(r: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit axis and angle in [0, pi] of a rotation matrix.

    The matrix logarithm of a rotation: it inverts Rodrigues'
    formula.

    R = I + sin(t) [w] + (1 - cos(t)) [w]^2 has three kinds of parts.
    The identity and [w]^2 are symmetric, [w] is antisymmetric. So
    - R - R.T keeps only the antisymmetric part: 2 sin(t) [w]
    - the trace keeps only the diagonal: 1 + 2 cos(t)
    That gives sin(t) * w and cos(t).
    Two things break this: sin(t) = 0, which happens at t = 0 and t = pi.

    Args:
        r (np.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple[np.ndarray, float]: (axis, angle).
    """
    r11, r12, r13 = r[0]
    r21, r22, r23 = r[1]
    r31, r32, r33 = r[2]

    cos_angle = (r11 + r22 + r33 - 1) / 2

    # The three independent entries of R - R.T are 2 sin(t) times the axis .
    axis_times_2sin = np.array([r32 - r23, r13 - r31, r21 - r12])
    sin_angle = np.linalg.norm(axis_times_2sin) / 2

    # (a) R = I: angle 0, axis undefined.
    if sin_angle == 0 and cos_angle > 0:
        return np.array([1.0, 0.0, 0.0]), 0.0

    # (b) angle pi: sin(t) ~ 0 and cos(t) ~ -1, so R - R.T carries no axis.
    if cos_angle < 0 and sin_angle < SIN_TOL:
        s = (r + r.T) / 2
        i = np.argmax(np.diag(s))
        if i == 2:
            axis = np.array([s[0, 2], s[1, 2], 1 + s[2, 2]]) / np.sqrt(
                2 * (1 + s[2, 2])
            )
        elif i == 1:
            axis = np.array([s[0, 1], 1 + s[1, 1], s[2, 1]]) / np.sqrt(
                2 * (1 + s[1, 1])
            )
        else:
            axis = np.array([1 + s[0, 0], s[1, 0], s[2, 0]]) / np.sqrt(
                2 * (1 + s[0, 0])
            )
        if np.dot(axis, axis_times_2sin) < 0:
            axis = -axis
        return axis, float(np.arctan2(sin_angle, cos_angle))

    # (c) otherwise: axis from (3.53)/(3.61), angle from sin and cos.
    angle = np.arctan2(sin_angle, cos_angle)
    return axis_times_2sin / (2 * sin_angle), float(angle)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return `q` scaled to unit length (xyzw).

    Boundary helper for noisy sources: floats drift off |q| = 1.

    Args:
        q (np.ndarray): Quaternion, shape (4,), not zero.

    Returns:
        np.ndarray: q / |q|.

    Raises:
        ValueError: If q is the zero quaternion (it has no direction).
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("cannot normalize the zero quaternion")
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of `q` (xyzw). For a unit quaternion this is the inverse.

    Args:
        q (np.ndarray): Quaternion, shape (4,).

    Returns:
        np.ndarray: (-x, -y, -z, w). It rotates by the same angle about the
            reversed axis, which undoes q.
    """
    # axis -> -axis in (axis sin(a/2), cos(a/2)) is the rotation by -angle,
    # because sin is odd and cos is even.
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 (xyzw): rotate by q2 first, then by q1.

    Same order as matrices: R(q1 * q2) = R(q1) @ R(q2).

    Args:
        q1 (np.ndarray): Quaternion, shape (4,).
        q2 (np.ndarray): Quaternion, shape (4,).

    Returns:
        np.ndarray: The product, shape (4,). Not commutative.
    """
    v1, s1 = q1[:3], q1[3]
    v2, s2 = q2[:3], q2[3]
    # The cross product is the only term that changes sign when the order
    # swaps, which is why q1 * q2 != q2 * q1.
    scalar = s1 * s2 - np.dot(v1, v2)
    vector = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return np.append(vector, scalar)


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit quaternion (xyzw) for a rotation of `angle` radians about unit `axis`.

    Args:
        axis (np.ndarray): Unit vector, shape (3,).
        angle (float): Radians.

    Returns:
        np.ndarray: (axis * sin(angle/2), cos(angle/2)).
    """
    return np.append(np.asarray(axis) * np.sin(angle / 2), np.cos(angle / 2))


def quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit axis and angle in [0, pi] of a unit quaternion (xyzw).

    Args:
        q (np.ndarray): Unit quaternion, shape (4,).

    Returns:
        tuple[np.ndarray, float]: (axis, angle). At angle 0 the axis is
            undefined and x is returned, like matrix_to_axis_angle.
    """
    if q[3] < 0:
        q = -q
    sin_half = np.linalg.norm(q[:3])
    cos_half = q[3]
    # arctan2, not 2*arccos(cos_half): arccos is flat near cos = 1, so a tiny
    # angle would lose its digits. sin_half is exact there.
    angle = 2 * np.arctan2(sin_half, cos_half)
    if sin_half == 0:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return q[:3] / sin_half, float(angle)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion (xyzw). The exit-test function.

    Args:
        q (np.ndarray): Unit quaternion, shape (4,). Not checked (boundary rule).

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    # Rodrigues is R = I + sin(a)[w] + (1 - cos(a))[w]^2. With v = w sin(a/2)
    # and s = cos(a/2):
    #   sin(a)     = 2 sin(a/2) cos(a/2)  so  sin(a)[w]        = 2 s [v]
    #   1 - cos(a) = 2 sin^2(a/2)         so  (1 - cos(a))[w]^2 = 2 [v]^2
    # which gives R = I + 2 s [v] + 2 [v]^2, written out entry by entry below.
    # Each entry is a product of two components of q: the 2[v]^2
    # term gives the diagonal 1 - 2(y^2 + z^2) etc. and the symmetric
    # off-diagonal part, 2 s [v] the antisymmetric part.
    x, y, z, w = q
    r11 = 1 - 2 * (y * y + z * z)
    r12 = 2 * (x * y - z * w)
    r13 = 2 * (x * z + y * w)
    r21 = 2 * (x * y + z * w)
    r22 = 1 - 2 * (x * x + z * z)
    r23 = 2 * (y * z - x * w)
    r31 = 2 * (x * z - y * w)
    r32 = 2 * (y * z + x * w)
    r33 = 1 - 2 * (x * x + y * y)
    return np.array(
        [
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33],
        ]
    )


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Unit quaternion (xyzw, scalar >= 0) of a rotation matrix.

    Args:
        r (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: Quaternion, shape (4,).
    """
    axis, angle = matrix_to_axis_angle(r)
    return axis_angle_to_quat(axis, angle)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation from `q0` (t=0) to `q1` (t=1), xyzw.

    Moves along the shortest arc at constant angular speed. Plain linear
    interpolation of the components (lerp) takes the same path but speeds up
    in the middle, and it is not unit length.

    Args:
        q0 (np.ndarray): Unit quaternion, shape (4,).
        q1 (np.ndarray): Unit quaternion, shape (4,).
        t (float): 0 to 1.

    Returns:
        np.ndarray: Unit quaternion, shape (4,).
    """
    dot = np.dot(q0, q1)
    # q1 and -q1 are the same rotation; take the one on the short side.
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > SLERP_TOL:
        return quat_normalize((1 - t) * q0 + t * q1)
    theta = np.arccos(dot)  # dot < SLERP_TOL, so arccos is not in its flat part
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)

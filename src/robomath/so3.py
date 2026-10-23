"""Robomath - the Lie group SO(3) and its algebra so(3) (Sola et al., micro Lie theory).

Equation numbers are from arXiv:1812.01537v9. Vectors in R^3 are rotation vectors:
axis * angle, in radians. Every function accepts leading batch dimensions: vectors
are (..., 3), matrices (..., 3, 3), and a single input is the case with none.
"""

import math

import numpy as np

from ._util import norm
from .rotations import inverse

SIN_TOL = 1e-5
# Below this |theta| the Jacobian coefficients use their Taylor series: the closed
# forms divide by t^2 and t^3 and lose about eps/t of the result there.
SMALL_THETA = 1e-3

# Four spaces to keep apart (paper Fig. 6):
#   R^3     rotation vector v          (what you store and pass around)
#   so(3)   skew matrix [v]x           (the algebra; what hat() builds)
#   SO(3)   rotation matrix R          (the group)
# hat/vee move between R^3 and so(3). exp/log move between so(3) and SO(3).
# The capitalized Exp/Log skip the middle step: R^3 <-> SO(3) directly.
# Python names: Exp -> exp_vec, Log -> log_vec, exp -> exp_so3, log -> log_so3.


def _rodrigues(x: float, y: float, z: float, k1: float, k2: float) -> np.ndarray:
    """I + k1 [v]x + k2 [v]x^2 for one vector v = (x, y, z) of Python floats.

    The single-vector fast path of exp_vec, jr and jr_inv: plain float arithmetic
    is several times faster than a handful of tiny NumPy calls.
    """
    xy, xz, yz = x * y, x * z, y * z
    return np.array(
        [
            [1 - k2 * (y * y + z * z), k2 * xy - k1 * z, k2 * xz + k1 * y],
            [k2 * xy + k1 * z, 1 - k2 * (x * x + z * z), k2 * yz - k1 * x],
            [k2 * xz - k1 * y, k2 * yz + k1 * x, 1 - k2 * (x * x + y * y)],
        ]
    )


def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]x of a 3-vector (paper eq. 10, Example 3).

    Args:
        v (np.ndarray): Rotation vectors, shape (..., 3).

    Returns:
        np.ndarray: The matrices [v]x, shape (..., 3, 3), so that
            [v]x @ u == np.cross(v, u).
    """
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        x, y, z = v.tolist()
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    m = np.zeros(v.shape[:-1] + (3, 3))
    m[..., 0, 1] = -v[..., 2]
    m[..., 0, 2] = v[..., 1]
    m[..., 1, 0] = v[..., 2]
    m[..., 1, 2] = -v[..., 0]
    m[..., 2, 0] = -v[..., 1]
    m[..., 2, 1] = v[..., 0]
    return m


def vee(m: np.ndarray) -> np.ndarray:
    """3-vector of a skew-symmetric matrix (paper eq. 11). Inverse of hat.

    Args:
        m (np.ndarray): Skew-symmetric matrices, shape (..., 3, 3).

    Returns:
        np.ndarray: Vectors v, shape (..., 3), with hat(v) == m.
    """
    m = np.asarray(m)
    if m.ndim == 2:
        return np.array([m[2, 1], m[0, 2], m[1, 0]], dtype=float)
    v = np.empty(m.shape[:-1])
    v[..., 0] = m[..., 2, 1]
    v[..., 1] = m[..., 0, 2]
    v[..., 2] = m[..., 1, 0]
    return v


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for a rotation of `angle` radians about unit `axis`.

    Args:
        axis (np.ndarray): Unit vectors, shape (..., 3).
        angle (float): Radians, scalar or shape (...); broadcasts with axis.

    Returns:
        np.ndarray: The rotation matrices, shape (..., 3, 3).
    """
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)
    if axis.ndim == 1 and angle.ndim == 0:
        x, y, z = axis.tolist()
        a = float(angle)
        return _rodrigues(x, y, z, math.sin(a), 1 - math.cos(a))

    # Construct the skew symmetric matrix
    w = hat(axis)
    angle = angle[..., None, None]

    # Rodrigues' rotation formula
    return np.eye(3) + np.sin(angle) * w + (1 - np.cos(angle)) * (w @ w)


def nearest_rotation(a: np.ndarray) -> np.ndarray:
    """Rotation matrix closest to a 3x3 matrix (orthogonal Procrustes problem).

    Use it where a matrix enters from a noisy source, or after many products,
    when rounding has made it slightly non-orthogonal.

    Args:
        a (np.ndarray): Matrices close to a rotation, shape (..., 3, 3).

    Returns:
        np.ndarray: The rotation matrices R (R.T @ R = I, det R = +1), shape
            (..., 3, 3), that minimize the Frobenius norm of a - R.
    """
    u, _, vt = np.linalg.svd(np.asarray(a, dtype=float))
    # U V^T is the closest orthogonal matrix, but it may be a reflection
    # (det = -1). Flipping the direction of the smallest singular value, the
    # last column of U, costs the least and gives a rotation.
    d = np.linalg.det(u @ vt)
    u[..., :, 2] *= d[..., None]
    return u @ vt


def _axis_at_pi(r: np.ndarray, axis_times_2sin: np.ndarray) -> np.ndarray:
    """Unit axes of rotations near pi (eq. 3.58-3.60); sign set by R - R.T."""
    s = (r + np.swapaxes(r, -1, -2)) / 2
    diag = np.diagonal(s, axis1=-2, axis2=-1)
    # Equations 3.58-3.60 are one formula: column i of S + I over sqrt(2 (1 + S_ii)),
    # with i the largest diagonal entry, which keeps the division away from zero.
    i = np.argmax(diag, axis=-1)
    column = np.take_along_axis(s, i[..., None, None], axis=-1)[..., 0] + np.eye(3)[i]
    axis = column / np.sqrt(2 * (1 + diag.max(axis=-1)))[..., None]
    flip = np.sum(axis * axis_times_2sin, axis=-1, keepdims=True) < 0
    return np.where(flip, -axis, axis)


def _matrix_to_axis_angle_single(r: np.ndarray) -> tuple[np.ndarray, float]:
    """matrix_to_axis_angle for one matrix, in plain float arithmetic."""
    (r11, r12, r13), (r21, r22, r23), (r31, r32, r33) = r.tolist()
    cos_angle = (r11 + r22 + r33 - 1) / 2
    vx, vy, vz = r32 - r23, r13 - r31, r21 - r12
    sin_angle = math.sqrt(vx * vx + vy * vy + vz * vz) / 2
    angle = math.atan2(sin_angle, cos_angle)
    if sin_angle == 0 and cos_angle > 0:
        return np.array([1.0, 0.0, 0.0]), angle
    if cos_angle < 0 and sin_angle < SIN_TOL:
        v = np.array([vx, vy, vz])
        return _axis_at_pi(r[None], v[None])[0], angle
    d = 2 * sin_angle
    return np.array([vx / d, vy / d, vz / d]), angle


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
        r (np.ndarray): Rotation matrices, shape (..., 3, 3).

    Returns:
        tuple[np.ndarray, float]: (axis, angle), shapes (..., 3) and (...).
    """
    r = np.asarray(r, dtype=float)
    if r.ndim == 2:
        return _matrix_to_axis_angle_single(r)
    cos_angle = (r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1) / 2

    # The three independent entries of R - R.T are 2 sin(t) times the axis .
    axis_times_2sin = np.empty(r.shape[:-1])
    axis_times_2sin[..., 0] = r[..., 2, 1] - r[..., 1, 2]
    axis_times_2sin[..., 1] = r[..., 0, 2] - r[..., 2, 0]
    axis_times_2sin[..., 2] = r[..., 1, 0] - r[..., 0, 1]
    sin_angle = norm(axis_times_2sin) / 2

    # The angle is the same in every case, and 0 at R = I and pi at a half turn.
    angle = np.arctan2(sin_angle, cos_angle)

    # (c) otherwise: axis from (3.53)/(3.61). A zero sin(t) is replaced by 1 in
    # the division; those entries are overwritten by (a) and (b) below.
    axis = axis_times_2sin / np.where(sin_angle > 0, 2 * sin_angle, 1.0)[..., None]

    # (a) R = I: angle 0, axis undefined.
    identity = (sin_angle == 0) & (cos_angle > 0)
    if identity.any():
        axis[identity] = [1.0, 0.0, 0.0]

    # (b) angle pi: sin(t) ~ 0 and cos(t) ~ -1, so R - R.T carries no axis.
    half_turn = (cos_angle < 0) & (sin_angle < SIN_TOL)
    if half_turn.any():
        axis[half_turn] = _axis_at_pi(r[half_turn], axis_times_2sin[half_turn])

    return axis, angle


def exp_vec(v: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector: the paper's Exp (eq. 21, 134).

    Args:
        v (np.ndarray): Rotation vectors, shape (..., 3). The length is the angle.

    Returns:
        np.ndarray: The rotation matrices, shape (..., 3, 3).
    """
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        x, y, z = v.tolist()
        t = math.sqrt(x * x + y * y + z * z)
        if t > 0:
            return _rodrigues(x, y, z, math.sin(t) / t, 2 * (math.sin(t / 2) / t) ** 2)
        return _rodrigues(x, y, z, 1.0, 0.5)
    t = norm(v)
    w = hat(v)
    # Rodrigues with the axis folded into v: I + sin(t)/t [v]x + (1-cos t)/t^2 [v]x^2.
    s, a = _sin_and_cos_ratios(t)
    return np.eye(3) + s[..., None, None] * w + a[..., None, None] * (w @ w)


def log_vec(r: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix: the paper's Log (eq. 22, 135).

    Args:
        r (np.ndarray): Rotation matrices, shape (..., 3, 3).

    Returns:
        np.ndarray: Rotation vectors, shape (..., 3), with length in [0, pi].
    """
    axis, angle = matrix_to_axis_angle(r)
    return axis * np.asarray(angle)[..., None]


def exp_so3(m: np.ndarray) -> np.ndarray:
    """Matrix exponential of an so(3) matrix: the paper's exp (eq. 23).

    Args:
        m (np.ndarray): Skew-symmetric matrix [v]x, shape (3, 3).

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    return exp_vec(vee(m))


def log_so3(r: np.ndarray) -> np.ndarray:
    """Matrix logarithm of a rotation matrix: the paper's log (eq. 24).

    Args:
        r (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: Skew-symmetric matrix [v]x, shape (3, 3).
    """
    return hat(log_vec(r))


def rplus(r: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Right plus R (+) tau: turn R by tau in R's own frame (eq. 25).

    Args:
        r (np.ndarray): A 3x3 rotation matrix.
        tau (np.ndarray): Small rotation vector in the local frame, shape (3,).

    Returns:
        np.ndarray: R @ Exp(tau).
    """
    return r @ exp_vec(tau)


def rminus(r2: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """Right minus R2 (-) R1: the local rotation vector from R1 to R2 (eq. 26).

    Args:
        r2 (np.ndarray): A 3x3 rotation matrix.
        r1 (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: Rotation vector tau, shape (3,), with rplus(r1, tau) == r2.
    """
    return log_vec(inverse(r1) @ r2)


def lplus(tau: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Left plus tau (+) R: turn R by tau in the world frame (eq. 27).

    Args:
        tau (np.ndarray): Small rotation vector in the global frame, shape (3,).
        r (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: Exp(tau) @ R.
    """
    return exp_vec(tau) @ r


def lminus(r2: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """Left minus R2 (-) R1: the global rotation vector from R1 to R2 (eq. 28).

    Args:
        r2 (np.ndarray): A 3x3 rotation matrix.
        r1 (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: Rotation vector tau, shape (3,), with lplus(tau, r1) == r2.
    """
    return log_vec(r2 @ inverse(r1))


def _sin_and_cos_ratios(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sin(t) / t and (1 - cos t) / t^2, exact for every t >= 0.

    1 - cos t = 2 sin^2(t/2) has no cancellation, so no series is needed. Only
    t = 0 uses its limits, 1 and 1/2.
    """
    positive = t > 0
    safe = np.where(positive, t, 1.0)  # keeps the unused branch from dividing by 0
    s = np.where(positive, np.sin(safe) / safe, 1.0)
    a = np.where(positive, 2 * (np.sin(safe / 2) / safe) ** 2, 0.5)
    return s, a


def _jr_coefficients(t: float) -> tuple[float, float]:
    """Coefficients (a, b) of jr: (1 - cos t) / t^2 and (t - sin t) / t^3."""
    t = np.asarray(t, dtype=float)
    small = t < SMALL_THETA
    safe = np.where(small, 1.0, t)  # keeps the unused branch from dividing by 0
    a = _sin_and_cos_ratios(t)[1]
    b = np.where(small, 1 / 6 - t**2 / 120, (safe - np.sin(safe)) / safe**3)
    return a, b


def _jr_inv_coefficient(t: float) -> float:
    """Coefficient c of jr_inv: 1 / t^2 - (1 + cos t) / (2 t sin t).

    Written with (1 + cos t) / sin t = 1 / tan(t/2), which has no cancellation
    near t = pi.
    """
    t = np.asarray(t, dtype=float)
    small = t < SMALL_THETA
    safe = np.where(small, 1.0, t)
    return np.where(
        small,
        1 / 12 + t**2 / 720,
        1 / safe**2 - 1 / (2 * safe * np.tan(safe / 2)),
    )


def jr(theta: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3) at a rotation vector (eq. 67, closed form 143).

    Maps a small change of the vector theta to a small turn in the local frame
    at Exp(theta): Exp(theta + d) ~ Exp(theta) Exp(jr(theta) @ d)   (eq. 68).

    Args:
        theta (np.ndarray): Rotation vectors, shape (..., 3).

    Returns:
        np.ndarray: The matrices Jr(theta), shape (..., 3, 3).
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        x, y, z = theta.tolist()
        t = math.sqrt(x * x + y * y + z * z)
        a = 2 * (math.sin(t / 2) / t) ** 2 if t > 0 else 0.5
        b = (t - math.sin(t)) / t**3 if t >= SMALL_THETA else 1 / 6 - t**2 / 120
        return _rodrigues(x, y, z, -a, b)
    a, b = _jr_coefficients(norm(theta))
    w = hat(theta)
    return np.eye(3) - a[..., None, None] * w + b[..., None, None] * (w @ w)


def jr_inv(theta: np.ndarray) -> np.ndarray:
    """Inverse of the right Jacobian (eq. 144). Gives Log(Exp(t) Exp(d)) (eq. 70).

    Args:
        theta (np.ndarray): Rotation vectors, shape (..., 3), with |theta| < 2 pi.

    Returns:
        np.ndarray: The matrices Jr(theta)^-1, shape (..., 3, 3).
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        x, y, z = theta.tolist()
        t = math.sqrt(x * x + y * y + z * z)
        if t >= SMALL_THETA:
            c = 1 / t**2 - 1 / (2 * t * math.tan(t / 2))
        else:
            c = 1 / 12 + t**2 / 720
        return _rodrigues(x, y, z, 0.5, c)
    c = _jr_inv_coefficient(norm(theta))
    w = hat(theta)
    return np.eye(3) + w / 2 + c[..., None, None] * (w @ w)


def jl(theta: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3) (eq. 71, closed form 145).

    Same as jr, but the turn is in the global frame:
    Exp(theta + d) ~ Exp(jl(theta) @ d) Exp(theta)   (eq. 72).

    Args:
        theta (np.ndarray): Rotation vectors, shape (..., 3).

    Returns:
        np.ndarray: The matrices Jl(theta), shape (..., 3, 3).
    """
    return np.swapaxes(jr(theta), -1, -2)


def jl_inv(theta: np.ndarray) -> np.ndarray:
    """Inverse of the left Jacobian (eq. 146). Gives Log(Exp(d) Exp(t)) (eq. 74).

    Args:
        theta (np.ndarray): Rotation vectors, shape (..., 3), with |theta| < 2 pi.

    Returns:
        np.ndarray: The matrices Jl(theta)^-1, shape (..., 3, 3).
    """
    return np.swapaxes(jr_inv(theta), -1, -2)

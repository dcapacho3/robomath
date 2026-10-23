"""Robomath - the Lie group SO(3) and its algebra so(3) (Sola et al., micro Lie theory).

Equation numbers are from arXiv:1812.01537v9. Vectors in R^3 are rotation vectors:
axis * angle, in radians.
"""

import numpy as np

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


def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]x of a 3-vector (paper eq. 10, Example 3).

    Args:
        v (np.ndarray): Rotation vector, shape (3,).

    Returns:
        np.ndarray: The 3x3 matrix [v]x, so that [v]x @ u == np.cross(v, u).
    """
    wx, wy, wz = v
    return np.array(
        [
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0],
        ]
    )


def vee(m: np.ndarray) -> np.ndarray:
    """3-vector of a skew-symmetric matrix (paper eq. 11). Inverse of hat.

    Args:
        m (np.ndarray): Skew-symmetric matrix, shape (3, 3).

    Returns:
        np.ndarray: Vector v, shape (3,), with hat(v) == m.
    """
    wx = m[2, 1]
    wy = m[0, 2]
    wz = m[1, 0]

    return np.array([wx, wy, wz])


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for a rotation of `angle` radians about unit `axis`.

    Args:
        axis (np.ndarray): Unit vector, shape (3,).
        angle (float): Radians.

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    # Construct the skew symmetric matrix
    w = hat(axis)

    # Rodrigues' rotation formula
    rot = np.eye(3) + np.sin(angle) * w + (1 - np.cos(angle)) * (w @ w)

    return rot


def _axis_at_pi(r: np.ndarray, axis_times_2sin: np.ndarray) -> np.ndarray:
    """Unit axis of a rotation near pi (eq. 3.58-3.60); sign set by R - R.T."""
    s = (r + r.T) / 2
    i = np.argmax(np.diag(s))
    if i == 2:
        axis = np.array([s[0, 2], s[1, 2], 1 + s[2, 2]]) / np.sqrt(2 * (1 + s[2, 2]))
    elif i == 1:
        axis = np.array([s[0, 1], 1 + s[1, 1], s[2, 1]]) / np.sqrt(2 * (1 + s[1, 1]))
    else:
        axis = np.array([1 + s[0, 0], s[1, 0], s[2, 0]]) / np.sqrt(2 * (1 + s[0, 0]))
    if np.dot(axis, axis_times_2sin) < 0:
        axis = -axis
    return axis


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
        axis = _axis_at_pi(r, axis_times_2sin)
        return axis, float(np.arctan2(sin_angle, cos_angle))

    # (c) otherwise: axis from (3.53)/(3.61), angle from sin and cos.
    angle = np.arctan2(sin_angle, cos_angle)
    return axis_times_2sin / (2 * sin_angle), float(angle)


def exp_vec(v: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector: the paper's Exp (eq. 21, 134).

    Args:
        v (np.ndarray): Rotation vector, shape (3,). Its length is the angle.

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    angle = np.linalg.norm(v)
    if angle == 0:
        return np.eye(3)
    axis = v / angle
    return axis_angle_to_matrix(axis, angle)


def log_vec(r: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix: the paper's Log (eq. 22, 135).

    Args:
        r (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: Rotation vector, shape (3,), with length in [0, pi].
    """
    axis, angle = matrix_to_axis_angle(r)
    return axis * angle


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


def _jr_coefficients(t: float) -> tuple[float, float]:
    """Coefficients (a, b) of jr: (1 - cos t) / t^2 and (t - sin t) / t^3."""
    if t < SMALL_THETA:
        return 1 / 2 - t**2 / 24, 1 / 6 - t**2 / 120
    return (1 - np.cos(t)) / t**2, (t - np.sin(t)) / t**3


def _jr_inv_coefficient(t: float) -> float:
    """Coefficient c of jr_inv: 1 / t^2 - (1 + cos t) / (2 t sin t)."""
    if t < SMALL_THETA:
        return 1 / 12 + t**2 / 720
    return 1 / t**2 - (1 + np.cos(t)) / (2 * t * np.sin(t))


def jr(theta: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3) at a rotation vector (eq. 67, closed form 143).

    Maps a small change of the vector theta to a small turn in the local frame
    at Exp(theta): Exp(theta + d) ~ Exp(theta) Exp(jr(theta) @ d)   (eq. 68).

    Args:
        theta (np.ndarray): Rotation vector, shape (3,).

    Returns:
        np.ndarray: The 3x3 matrix Jr(theta).
    """
    a, b = _jr_coefficients(np.linalg.norm(theta))
    w = hat(theta)
    return np.eye(3) - a * w + b * (w @ w)


def jr_inv(theta: np.ndarray) -> np.ndarray:
    """Inverse of the right Jacobian (eq. 144). Gives Log(Exp(t) Exp(d)) (eq. 70).

    Args:
        theta (np.ndarray): Rotation vector, shape (3,), with |theta| < 2 pi.

    Returns:
        np.ndarray: The 3x3 matrix Jr(theta)^-1.
    """
    c = _jr_inv_coefficient(np.linalg.norm(theta))
    w = hat(theta)
    return np.eye(3) + w / 2 + c * (w @ w)


def jl(theta: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3) (eq. 71, closed form 145).

    Same as jr, but the turn is in the global frame:
    Exp(theta + d) ~ Exp(jl(theta) @ d) Exp(theta)   (eq. 72).

    Args:
        theta (np.ndarray): Rotation vector, shape (3,).

    Returns:
        np.ndarray: The 3x3 matrix Jl(theta).
    """
    return jr(theta).T


def jl_inv(theta: np.ndarray) -> np.ndarray:
    """Inverse of the left Jacobian (eq. 146). Gives Log(Exp(d) Exp(t)) (eq. 74).

    Args:
        theta (np.ndarray): Rotation vector, shape (3,), with |theta| < 2 pi.

    Returns:
        np.ndarray: The 3x3 matrix Jl(theta)^-1.
    """
    return jr_inv(theta).T

"""Robomath - Property-based tests: laws that hold for every input, not just samples.

Hypothesis generates the inputs, including awkward ones (tiny, huge, near pi, near
gimbal lock), and shrinks a failure to the smallest case. Each test states a law.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

import robomath as rm

# The first call of a function can be slow; a time limit per example only adds noise.
settings.register_profile("robomath", deadline=None)
settings.load_profile("robomath")

TOL = 1e-9
PI = np.pi


def floats(limit):
    """Finite floats in [-limit, limit]."""
    return st.floats(-limit, limit, allow_nan=False, allow_infinity=False)


def vectors(size, limit=10.0):
    """Arrays of `size` finite floats."""
    return arrays(np.float64, size, elements=floats(limit))


def direction(z, phi):
    """Unit vector from its height z in [-1, 1] and its angle phi about the z axis."""
    ring = np.sqrt(1.0 - z**2)
    return np.array([ring * np.cos(phi), ring * np.sin(phi), z])


def rotvecs(max_angle=3.1):
    """Rotation vectors with length in [0, max_angle], below pi by default."""
    return st.builds(
        lambda z, phi, angle: direction(z, phi) * angle,
        st.floats(-1, 1),
        st.floats(0, 2 * PI),
        st.floats(0, max_angle),
    )


rotations = vectors(3).map(rm.exp_vec)  # any rotation, from a vector of any length
# A unit quaternion from three angles (hyperspherical coordinates): no input is
# thrown away, so Hypothesis never has to reject a case.
quats = st.builds(
    lambda a, b, c: np.array(
        [
            np.sin(a) * np.sin(b) * np.cos(c),
            np.sin(a) * np.sin(b) * np.sin(c),
            np.sin(a) * np.cos(b),
            np.cos(a),
        ]
    ),
    st.floats(0, PI),
    st.floats(0, PI),
    st.floats(0, 2 * PI),
)
taus = st.tuples(vectors(3), rotvecs()).map(lambda p: rm.twist(*p))
poses = taus.map(rm.exp_vec_se3)


def is_rotation(r):
    """Orthogonal with determinant +1."""
    assert_allclose(r @ r.T, np.eye(3), rtol=0, atol=TOL)
    assert_allclose(np.linalg.det(r), 1.0, rtol=0, atol=TOL)


@given(vectors(3))
def test_exp_gives_a_rotation(v):
    """Exp of any vector is a rotation matrix."""
    is_rotation(rm.exp_vec(v))


@given(vectors(3), vectors(3))
def test_hat_is_the_cross_product_and_vee_undoes_it(v, u):
    """hat(v) u == v x u, and vee(hat(v)) == v."""
    assert_allclose(rm.hat(v) @ u, np.cross(v, u), rtol=0, atol=TOL)
    assert_allclose(rm.vee(rm.hat(v)), v, rtol=0, atol=0)


@given(vectors(6))
def test_se3_vee_undoes_hat(tau):
    """vee_se3(hat_se3(tau)) == tau."""
    assert_allclose(rm.vee_se3(rm.hat_se3(tau)), tau, rtol=0, atol=0)


@given(rotvecs())
def test_log_inverts_exp_below_pi(v):
    """Log(Exp(v)) == v when |v| < pi."""
    assert_allclose(rm.log_vec(rm.exp_vec(v)), v, rtol=0, atol=TOL)


@given(rotations)
def test_exp_inverts_log(r):
    """Exp(Log(R)) == R for every rotation, also when |v| was beyond pi."""
    assert_allclose(rm.exp_vec(rm.log_vec(r)), r, rtol=0, atol=TOL)


@given(rotations, rotations)
def test_rotation_inverse_is_transpose_and_composition_stays_a_rotation(r1, r2):
    """The inverse undoes R, and a product of rotations is a rotation."""
    assert_allclose(rm.inverse(r1) @ r1, np.eye(3), rtol=0, atol=TOL)
    is_rotation(r1 @ r2)


@given(quats)
def test_quaternion_matrix_round_trip(q):
    """A quaternion and its negative give one rotation; the matrix round trip holds."""
    r = rm.quat_to_matrix(q)
    is_rotation(r)
    assert_allclose(rm.quat_to_matrix(-q), r, rtol=0, atol=TOL)
    assert_allclose(rm.quat_to_matrix(rm.matrix_to_quat(r)), r, rtol=0, atol=TOL)


@given(quats, quats)
def test_quaternion_product_is_the_matrix_product(q1, q2):
    """The product of quaternions is the product of the matrices."""
    expected = rm.quat_to_matrix(q1) @ rm.quat_to_matrix(q2)
    assert_allclose(
        rm.quat_to_matrix(rm.quat_multiply(q1, q2)), expected, rtol=0, atol=TOL
    )


@given(quats, quats, st.floats(0, 1))
def test_slerp_stays_a_unit_quaternion_and_hits_its_ends(q1, q2, t):
    """Slerp is a unit quaternion, and t = 0 and t = 1 give the two rotations."""
    assert_allclose(np.linalg.norm(rm.slerp(q1, q2, t)), 1.0, rtol=0, atol=TOL)
    for end, q in ((0.0, q1), (1.0, q2)):
        got = rm.quat_to_matrix(rm.slerp(q1, q2, end))
        assert_allclose(got, rm.quat_to_matrix(q), rtol=0, atol=TOL)


@given(rotations)
def test_euler_round_trip(r):
    """Matrix -> ZYX Euler -> matrix gives the matrix back, near gimbal lock too."""
    back = rm.euler_zyx_to_matrix(*rm.matrix_to_euler_zyx(r))
    assert_allclose(back, r, rtol=0, atol=1e-6)


@given(taus)
def test_se3_log_inverts_exp(tau):
    """Log(Exp(tau)) == tau when the rotation part is below pi."""
    assert_allclose(rm.log_vec_se3(rm.exp_vec_se3(tau)), tau, rtol=0, atol=TOL)


@given(poses, poses)
def test_pose_inverse_and_adjoint(m1, m2):
    """M M^-1 = I, the inverse of a product reverses the order, Ad is a homomorphism."""
    assert_allclose(m1 @ rm.trans_inv(m1), np.eye(4), rtol=0, atol=TOL)
    assert_allclose(
        rm.trans_inv(m1 @ m2), rm.trans_inv(m2) @ rm.trans_inv(m1), rtol=0, atol=TOL
    )
    assert_allclose(
        rm.adjoint(m1 @ m2), rm.adjoint(m1) @ rm.adjoint(m2), rtol=0, atol=1e-7
    )


@given(poses, taus)
def test_right_and_left_plus_agree_through_the_adjoint(m, tau):
    """lplus(Ad(M) tau, M) == rplus(M, tau)."""
    left = rm.lplus_se3(rm.adjoint(m) @ tau, m)
    assert_allclose(left, rm.rplus_se3(m, tau), rtol=1e-6, atol=1e-6)


@given(rotvecs(PI), rotvecs(PI), rotvecs(PI))
def test_rotation_distance_is_a_metric(a, b, c):
    """The angle between rotations is symmetric and obeys the triangle inequality."""
    ra, rb, rc = rm.exp_vec(a), rm.exp_vec(b), rm.exp_vec(c)
    assert_allclose(rm.distance_so3(ra, rb), rm.distance_so3(rb, ra), rtol=0, atol=TOL)
    d = rm.distance_so3
    assert d(ra, rc) <= d(ra, rb) + d(rb, rc) + TOL


# Four points spanning space, so any extra points keep the fit well posed.
SPANNING = np.array([[0.0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]])


@given(rotations, vectors(3), arrays(np.float64, (4, 3), elements=floats(5.0)))
def test_kabsch_recovers_a_known_motion(r, t, extra):
    """Points moved by (R, t) are fitted back to the same R and t."""
    points = np.concatenate([SPANNING, extra])
    got_r, got_t = rm.kabsch(points, points @ r.T + t)
    assert_allclose(got_r, r, rtol=0, atol=1e-8)
    assert_allclose(got_t, t, rtol=0, atol=1e-8)


@given(taus, st.floats(-2, 2))
def test_sim3_log_inverts_exp_and_inverse(tau, sigma):
    """Sim(3): Log(Exp(tau)) == tau, and M M^-1 = I, for scales between e^-2 and e^2."""
    tau7 = np.append(tau, sigma)
    m = rm.exp_vec_sim3(tau7)
    assert_allclose(rm.log_vec_sim3(m), tau7, rtol=0, atol=1e-7)
    assert_allclose(m @ rm.sim3_inv(m), np.eye(4), rtol=0, atol=1e-8)


@given(taus, vectors(3))
def test_se23_log_inverts_exp_and_inverse(tau, nu):
    """SE_2(3): Log(Exp(tau)) == tau, and M M^-1 = I."""
    tau9 = np.append(tau, nu)
    m = rm.exp_vec_se23(tau9)
    assert_allclose(rm.log_vec_se23(m), tau9, rtol=0, atol=1e-7)
    assert_allclose(m @ rm.se23_inv(m), np.eye(5), rtol=0, atol=1e-8)


@given(
    arrays(np.float64, st.tuples(st.integers(0, 4), st.just(3)), elements=floats(10.0))
)
def test_batched_exp_equals_a_loop(vs):
    """exp_vec on a stack gives the stack of single calls, also for zero rows."""
    got = rm.exp_vec(vs)
    assert got.shape == (len(vs), 3, 3)
    for v, r in zip(vs, got, strict=True):
        assert_allclose(r, rm.exp_vec(v), rtol=0, atol=1e-12)

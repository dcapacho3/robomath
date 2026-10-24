"""Robomath - Every function gives, for a batch, what a loop of single calls gives.

Every function exported from robomath must accept leading dimensions: a rotation
vector of shape (..., 3), a rotation matrix of shape (..., 3, 3), and so on.
Functions of se3.py that are not exported yet are skipped until they are.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_rotations import TOL

import robomath as rm
from robomath import se3

BATCH_SHAPES = [(5,), (2, 3)]  # one leading dimension, and two

RNG = np.random.default_rng(0)


def _rotvec():
    return RNG.normal(size=3)


def _unit():
    v = RNG.normal(size=3)
    return v / np.linalg.norm(v)


def _rotmat():
    return rm.exp_vec(_rotvec())


def _pose():
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = _rotmat(), RNG.normal(size=3)
    return m


def _se3_matrix():
    m = np.zeros((4, 4))
    m[:3, :3], m[:3, 3] = rm.hat(_rotvec()), RNG.normal(size=3)
    return m


def _pose2():
    m = np.eye(3)
    m[:2, :2], m[:2, 2] = rm.exp_so2(RNG.uniform(-3, 3)), RNG.normal(size=2)
    return m


def _se2_matrix():
    m = np.zeros((3, 3))
    m[:2, 2] = RNG.normal(size=2)
    m[0, 1], m[1, 0] = -(a := RNG.normal()), a
    return m


def _cluster(make):
    """Four items close together, so their mean is unique."""
    return lambda: np.array([make(RNG.normal(size=3) * 0.2) for _ in range(4)])


# One generator per kind of argument. Each returns ONE sample.
GENERATORS = {
    "rotvec": _rotvec,
    "unit": _unit,
    "angle": lambda: RNG.uniform(-np.pi, np.pi),
    "fraction": lambda: RNG.uniform(0.1, 0.9),
    "skew": lambda: rm.hat(_rotvec()),
    "rotmat": _rotmat,
    "noisy": lambda: _rotmat() + 1e-3 * RNG.normal(size=(3, 3)),
    "quat": lambda: rm.matrix_to_quat(_rotmat()),
    "quat_raw": lambda: RNG.normal(size=4) * 3,  # not unit length
    "tau": lambda: RNG.normal(size=6),
    "pose": _pose,
    "se3_matrix": _se3_matrix,
    "rotset": _cluster(rm.exp_vec),
    "poseset": _cluster(lambda v: rm.exp_vec_se3(np.r_[RNG.normal(size=3), v])),
    "poseset2": _cluster(lambda v: rm.exp_vec_se2(np.r_[RNG.normal(size=2), v[2]])),
    "rot2": lambda: rm.exp_so2(RNG.uniform(-3, 3)),
    "tau2": lambda: RNG.normal(size=3),
    "pose2": _pose2,
    "se2_matrix": _se2_matrix,
}

# (function name, kinds of its arguments). Functions of SE(3) live in se3.
CASES = [
    ("hat", ["rotvec"]),
    ("vee", ["skew"]),
    ("exp_vec", ["rotvec"]),
    ("log_vec", ["rotmat"]),
    ("exp_so3", ["skew"]),
    ("log_so3", ["rotmat"]),
    ("rplus", ["rotmat", "rotvec"]),
    ("rminus", ["rotmat", "rotmat"]),
    ("lplus", ["rotvec", "rotmat"]),
    ("lminus", ["rotmat", "rotmat"]),
    ("jr", ["rotvec"]),
    ("jr_inv", ["rotvec"]),
    ("jl", ["rotvec"]),
    ("jl_inv", ["rotvec"]),
    ("axis_angle_to_matrix", ["unit", "angle"]),
    ("matrix_to_axis_angle", ["rotmat"]),
    ("nearest_rotation", ["noisy"]),
    ("rotx", ["angle"]),
    ("roty", ["angle"]),
    ("rotz", ["angle"]),
    ("inverse", ["rotmat"]),
    ("euler_zyx_to_matrix", ["angle", "angle", "angle"]),
    ("matrix_to_euler_zyx", ["rotmat"]),
    ("quat_normalize", ["quat_raw"]),
    ("quat_conjugate", ["quat"]),
    ("quat_multiply", ["quat", "quat"]),
    ("axis_angle_to_quat", ["unit", "angle"]),
    ("quat_to_axis_angle", ["quat"]),
    ("quat_to_matrix", ["quat"]),
    ("matrix_to_quat", ["rotmat"]),
    ("slerp", ["quat", "quat", "fraction"]),
    ("quat_to_euler_zyx", ["quat"]),
    ("euler_zyx_to_quat", ["angle", "angle", "angle"]),
    ("twist", ["rotvec", "rotvec"]),
    ("linear", ["tau"]),
    ("angular", ["tau"]),
    ("rp_to_trans", ["rotmat", "rotvec"]),
    ("trans_to_rp", ["pose"]),
    ("trans_inv", ["pose"]),
    ("hat_se3", ["tau"]),
    ("vee_se3", ["se3_matrix"]),
    ("exp_vec_se3", ["tau"]),
    ("log_vec_se3", ["pose"]),
    ("adjoint", ["pose"]),
    ("rplus_se3", ["pose", "tau"]),
    ("rminus_se3", ["pose", "pose"]),
    ("lplus_se3", ["tau", "pose"]),
    ("lminus_se3", ["pose", "pose"]),
    ("distance_so3", ["rotmat", "rotmat"]),
    ("interp_so3", ["rotmat", "rotmat", "fraction"]),
    ("interp_se3", ["pose", "pose", "fraction"]),
    ("mean_so3", ["rotset"]),
    ("mean_se3", ["poseset"]),
    ("interp_se2", ["pose2", "pose2", "fraction"]),
    ("mean_se2", ["poseset2"]),
    ("exp_so2", ["angle"]),
    ("log_so2", ["rot2"]),
    ("wrap_angle", ["angle"]),
    ("hat_se2", ["tau2"]),
    ("vee_se2", ["se2_matrix"]),
    ("exp_vec_se2", ["tau2"]),
    ("log_vec_se2", ["pose2"]),
    ("adjoint_se2", ["pose2"]),
    ("rplus_se2", ["pose2", "tau2"]),
    ("rminus_se2", ["pose2", "pose2"]),
    ("lplus_se2", ["tau2", "pose2"]),
    ("lminus_se2", ["pose2", "pose2"]),
]


def _function(name):
    return getattr(rm, name, None) or getattr(se3, name)


def _as_tuple(result):
    return result if isinstance(result, tuple) else (result,)


def _stack(samples, shape):
    """Stack one-per-call samples into an array with `shape` leading dimensions."""
    return np.array(samples).reshape(shape + np.shape(samples[0]))


@pytest.mark.parametrize("shape", BATCH_SHAPES)
@pytest.mark.parametrize("name, kinds", CASES)
def test_batch_matches_loop(name, kinds, shape):
    """f(batch) equals the stack of f(single) for each sample in the batch."""
    if not hasattr(rm, name):
        pytest.skip("not exported yet")
    func = _function(name)
    n = int(np.prod(shape))
    samples = [[GENERATORS[kind]() for kind in kinds] for _ in range(n)]
    args = [_stack([s[i] for s in samples], shape) for i in range(len(kinds))]
    batched = _as_tuple(func(*args))
    singles = [_as_tuple(func(*s)) for s in samples]
    for i, out in enumerate(batched):
        expected = _stack([s[i] for s in singles], shape)
        assert_allclose(out, expected, rtol=0, atol=TOL)


# Inputs that take the special branches of a function, mixed in one batch: the
# batch must pick the right branch for each row.
PI = np.pi
SPECIAL_ROTVECS = [
    np.zeros(3),
    np.array([1e-12, 0, 0]),
    np.array([0, 0, 0.9e-3]),
    np.array([0, 0, 1.1e-3]),
    np.array([0.3, 0.4, 0]),
    np.array([0, PI - 1e-7, 0]),
    np.array([1.0, 2.0, 3.0]),
]
SPECIAL_ROTMATS = [rm.exp_vec(v) for v in SPECIAL_ROTVECS] + [
    np.eye(3),
    rm.rotx(PI),
    rm.roty(PI),
    rm.axis_angle_to_matrix(np.array([1.0, 1.0, 0.0]) / np.sqrt(2), PI),
    rm.euler_zyx_to_matrix(0.3, PI / 2, 0.2),
    rm.euler_zyx_to_matrix(0.3, -PI / 2, 0.2),
]
SPECIAL_QUATS = [rm.matrix_to_quat(r) for r in SPECIAL_ROTMATS]
SPECIAL_QUATS += [-q for q in SPECIAL_QUATS[:4]] + [np.array([1.0, 0, 0, 0])]
# Pairs 1e-7 rad apart, which slerp interpolates with its lerp branch.
NEAR_FIRST = SPECIAL_QUATS[:6]
NEAR_SECOND = [
    rm.quat_multiply(rm.axis_angle_to_quat(np.array([0.0, 0.6, 0.8]), 1e-7), q)
    for q in NEAR_FIRST
]
MIXED = {
    "hat": [SPECIAL_ROTVECS],
    "exp_vec": [SPECIAL_ROTVECS],
    "jr": [SPECIAL_ROTVECS],
    "jr_inv": [SPECIAL_ROTVECS],
    "jl": [SPECIAL_ROTVECS],
    "jl_inv": [SPECIAL_ROTVECS],
    "log_vec": [SPECIAL_ROTMATS],
    "matrix_to_axis_angle": [SPECIAL_ROTMATS],
    "matrix_to_euler_zyx": [SPECIAL_ROTMATS],
    "matrix_to_quat": [SPECIAL_ROTMATS],
    "nearest_rotation": [SPECIAL_ROTMATS],
    "quat_to_matrix": [SPECIAL_QUATS],
    "quat_to_axis_angle": [SPECIAL_QUATS],
    "quat_to_euler_zyx": [SPECIAL_QUATS],
    "slerp": [
        SPECIAL_QUATS + NEAR_FIRST,
        SPECIAL_QUATS[::-1] + NEAR_SECOND,
        ([0.0, 0.3, 1.0] * 8)[: len(SPECIAL_QUATS) + len(NEAR_FIRST)],
    ],
}


@pytest.mark.parametrize("name", MIXED)
def test_batch_with_mixed_branches(name):
    """A batch holding special cases gives what one call per row gives."""
    func = getattr(rm, name)
    columns = MIXED[name]
    n = min(len(c) for c in columns)
    args = [np.array(c[:n]) for c in columns]
    batched = _as_tuple(func(*args))
    singles = [_as_tuple(func(*(c[i] for c in columns))) for i in range(n)]
    for k, out in enumerate(batched):
        expected = np.array([s[k] for s in singles])
        assert_allclose(out, expected, rtol=0, atol=TOL)


def test_mixed_batch_has_the_right_angles():
    """A mixed batch of rotations gives the known angles, not only a loop's."""
    batch = np.array([np.eye(3), rm.rotx(PI), rm.rotz(0.5), rm.roty(1e-9)])
    _, angle = rm.matrix_to_axis_angle(batch)
    assert_allclose(angle, [0.0, PI, 0.5, 1e-9], rtol=0, atol=1e-12)


def test_one_argument_batched_and_the_other_single():
    """A single rotation broadcasts against a batch of vectors, quaternions, times."""
    r = rm.exp_vec(np.array([0.1, 0.2, 0.3]))
    taus = np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3], [0.1, 0.1, 0.1]])
    assert rm.rplus(r, taus).shape == (4, 3, 3)
    assert rm.lplus(taus, r).shape == (4, 3, 3)
    q = rm.matrix_to_quat(r)
    qs = rm.matrix_to_quat(rm.exp_vec(taus))
    assert rm.quat_multiply(q, qs).shape == (4, 4)
    assert rm.slerp(q, qs, np.linspace(0, 1, 4)).shape == (4, 4)
    for i in range(4):
        assert_allclose(rm.rplus(r, taus)[i], rm.rplus(r, taus[i]), rtol=0, atol=TOL)


def test_empty_batch():
    """Zero rows in, zero rows out, with no warning."""
    assert rm.exp_vec(np.zeros((0, 3))).shape == (0, 3, 3)
    assert rm.matrix_to_quat(np.zeros((0, 3, 3))).shape == (0, 4)
    axis, angle = rm.matrix_to_axis_angle(np.zeros((0, 3, 3)))
    assert axis.shape == (0, 3)
    assert angle.shape == (0,)


@pytest.mark.parametrize("name", ["nearest_rotation", "matrix_to_axis_angle"])
def test_inputs_are_not_changed(name):
    """The functions write only to their own results."""
    x = np.array(SPECIAL_ROTMATS) + 1e-3
    before = x.copy()
    getattr(rm, name)(x)
    assert np.array_equal(x, before)


def test_results_are_float64():
    """Integer lists and float32 arrays in, float64 out."""
    assert rm.hat(np.array([1, 2, 3])).dtype == np.float64
    assert rm.exp_vec(np.array([0.1, 0.2, 0.3], dtype=np.float32)).dtype == np.float64
    assert rm.exp_vec([0.1, 0.2, 0.3]).shape == (3, 3)

"""Robomath - Test the Jacobians of SO(3) (Sola et al., section II-G and App. B-C)."""

# pylint: disable=protected-access

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from test_rotations import TOL

from robomath import (
    exp_vec,
    hat,
    jl,
    jl_inv,
    jr,
    jr_inv,
    log_vec,
    rminus,
    rplus,
    so3,
)
from robomath.so3 import SMALL_THETA

RNG = np.random.default_rng(0)
# Rotation vectors of length 0.1 .. 3.0: away from 0 (series) and pi (sin t = 0).
THETAS = RNG.normal(size=(30, 3))
THETAS = THETAS / np.linalg.norm(THETAS, axis=1, keepdims=True)
THETAS = THETAS * np.linspace(0.1, 3.0, 30)[:, None]
MATRICES = Rotation.random(30, rng=1).as_matrix()
H = 1e-6  # numerical-derivative step: error ~H (curve) + ~1e-16/H (rounding)
D = 1e-7 * np.array([1.0, -2.0, 0.5])  # a tiny change of theta
TAU = np.array([0.2, 0.1, -0.3])
POINT = np.array([0.3, -1.0, 2.0])
NUMERIC_TOL = 1e-5  # the numeric derivative is only good to about H


def numeric_right_jacobian(f, r, minus):
    """Right Jacobian Df(R)/DR of f at R, by the definition (eq. 41a).

    Column i is (f(R (+) H e_i) (-) f(R)) / H. `minus` is the difference in
    the space f lands in: rminus for rotation matrices, np.subtract for vectors.
    """
    columns = [minus(f(rplus(r, H * e)), f(r)) / H for e in np.eye(3)]
    return np.column_stack(columns)


def test_jacobians_at_zero_are_identity():
    """jr, jr_inv, jl and jl_inv of the zero vector are the identity."""
    for jacobian in (jr, jr_inv, jl, jl_inv):
        assert_allclose(jacobian(np.zeros(3)), np.eye(3), rtol=0, atol=TOL)


def test_series_and_closed_form_agree_at_the_cut_off():
    """Just below and just above SMALL_THETA the matrices agree (no jump)."""
    axis = np.array([0.6, 0.0, 0.8])
    below, above = axis * SMALL_THETA * (1 - 1e-9), axis * SMALL_THETA * (1 + 1e-9)
    for jacobian in (jr, jr_inv):
        assert_allclose(jacobian(below), jacobian(above), rtol=0, atol=1e-12)


@pytest.mark.parametrize("theta", THETAS)
def test_jr_is_the_derivative_of_exp(theta):
    """Exp(theta + d) ~ Exp(theta) Exp(jr(theta) @ d) for tiny d (eq. 68)."""
    assert_allclose(
        exp_vec(theta + D), exp_vec(theta) @ exp_vec(jr(theta) @ D), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("theta", THETAS)
def test_jr_inv_is_the_derivative_of_log(theta):
    """Log(Exp(theta) Exp(d)) ~ theta + jr_inv(theta) @ d for tiny d (eq. 70)."""
    assert_allclose(
        log_vec(exp_vec(theta) @ exp_vec(D)),
        theta + jr_inv(theta) @ D,
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("theta", THETAS)
def test_jl_is_the_left_derivative_of_exp(theta):
    """Exp(theta + d) ~ Exp(jl(theta) @ d) Exp(theta) for tiny d (eq. 72)."""
    assert_allclose(
        exp_vec(theta + D), exp_vec(jl(theta) @ D) @ exp_vec(theta), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("theta", THETAS)
def test_jl_inv_is_the_left_derivative_of_log(theta):
    """Log(Exp(d) Exp(theta)) ~ theta + jl_inv(theta) @ d for tiny d (eq. 74)."""
    assert_allclose(
        log_vec(exp_vec(D) @ exp_vec(theta)),
        theta + jl_inv(theta) @ D,
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("theta", THETAS)
def test_inverses(theta):
    """jr_inv @ jr == I and jl_inv @ jl == I."""
    assert_allclose(jr_inv(theta) @ jr(theta), np.eye(3), rtol=0, atol=1e-12)
    assert_allclose(jl_inv(theta) @ jl(theta), np.eye(3), rtol=0, atol=1e-12)


@pytest.mark.parametrize("theta", THETAS)
def test_jl_is_jr_transposed_and_jr_of_minus_theta(theta):
    """Jl == jr.T (eq. 147) and jl(theta) == jr(-theta) (eq. 76)."""
    assert_allclose(jl(theta), jr(theta).T, rtol=0, atol=TOL)
    assert_allclose(jl(theta), jr(-theta), rtol=0, atol=TOL)


@pytest.mark.parametrize("theta", THETAS)
def test_adjoint_from_jacobians(theta):
    """jl(theta) @ jr_inv(theta) == exp_vec(theta): Ad_R = R (eq. 75, 139)."""
    assert_allclose(jl(theta) @ jr_inv(theta), exp_vec(theta), rtol=0, atol=1e-12)


def test_series_matches_closed_form(monkeypatch):
    """The Taylor series equals the closed form at a length where both are exact.

    At |theta| < SMALL_THETA a wrong series constant changes the result by less
    than the other tests can see. Here the cut-off is moved so the series runs
    at a larger t, where a wrong constant shows.
    """
    t = 0.02
    monkeypatch.setattr(so3, "SMALL_THETA", 1.0)
    series = (*so3._jr_coefficients(t), so3._jr_inv_coefficient(t))
    monkeypatch.setattr(so3, "SMALL_THETA", 0.0)
    closed = (*so3._jr_coefficients(t), so3._jr_inv_coefficient(t))
    assert_allclose(series, closed, rtol=2e-9)
    # The first coefficient has no series; it must equal its plain definition.
    assert_allclose(closed[0], (1 - np.cos(t)) / t**2, rtol=1e-9)


@pytest.mark.parametrize("t", [1e-8, 1e-6, 1e-5, 1e-4])
def test_tiny_vectors_follow_the_low_order_expansion(t):
    """For tiny theta: jr = I - W/2 + W^2/6 and jr_inv = I + W/2 + W^2/12."""
    w = hat(t * np.array([0.6, 0.0, 0.8]))
    atol = 1e-15 + t**3 / 10  # the first term left out is of order t^3
    theta = t * np.array([0.6, 0.0, 0.8])
    assert_allclose(jr(theta), np.eye(3) - w / 2 + w @ w / 6, rtol=0, atol=atol)
    assert_allclose(jr_inv(theta), np.eye(3) + w / 2 + w @ w / 12, rtol=0, atol=atol)


# The blocks below use numeric_right_jacobian to check results the paper
# states (App. B-C). They need no library function: only the definition.


@pytest.mark.parametrize("r", MATRICES)
def test_jacobian_of_inversion(r):
    """D(R^-1)/DR == -R: f(R) = R.T, the output is a rotation (eq. 140)."""
    jacobian = numeric_right_jacobian(lambda x: x.T, r, rminus)
    assert_allclose(jacobian, -r, rtol=0, atol=NUMERIC_TOL)


@pytest.mark.parametrize("r", MATRICES)
def test_jacobian_of_rotating_a_point(r):
    """D(R p)/DR == -R [p]x: f(R) = R @ p, the output is a vector (paper II-G)."""
    jacobian = numeric_right_jacobian(lambda x: x @ POINT, r, np.subtract)
    assert_allclose(jacobian, -r @ hat(POINT), rtol=0, atol=NUMERIC_TOL)


@pytest.mark.parametrize("r", MATRICES)
def test_jacobian_of_plus_with_respect_to_r(r):
    """D(R (+) tau)/DR == Exp(tau)^T: composition block (eq. 148)."""
    jacobian = numeric_right_jacobian(lambda x: rplus(x, TAU), r, rminus)
    assert_allclose(jacobian, exp_vec(TAU).T, rtol=0, atol=NUMERIC_TOL)


@pytest.mark.parametrize("theta", THETAS)
def test_jacobian_of_plus_with_respect_to_tau(theta):
    """D(R (+) tau)/Dtau == jr(tau): the other block (eq. 148)."""
    r = MATRICES[0]  # the result does not depend on R
    columns = [rminus(rplus(r, theta + H * e), rplus(r, theta)) / H for e in np.eye(3)]
    assert_allclose(np.column_stack(columns), jr(theta), rtol=0, atol=NUMERIC_TOL)


@pytest.mark.parametrize("gap", [1e-3, 1e-6, 1e-7, 0.0, -1e-7])
def test_jr_inv_stays_accurate_near_pi(gap):
    """jr_inv @ jr is the identity at |theta| = pi - gap, on both sides of pi."""
    theta = np.array([0.6, 0.0, 0.8]) * (np.pi - gap)
    assert_allclose(jr_inv(theta) @ jr(theta), np.eye(3), rtol=0, atol=1e-13)


@pytest.mark.parametrize("jacobian", [jr, jr_inv])
@pytest.mark.parametrize("batched", [False, True])
def test_series_in_matrix_form_matches_closed_form(jacobian, batched, monkeypatch):
    """Single-vector and batch paths both use the right series constants.

    The cut-off is moved so the series runs at |theta| = 0.02, where a wrong
    constant shows; the closed form (cut-off 0) is the reference.
    """
    theta = np.array([0.012, 0.0, 0.016])

    def run():
        return jacobian(theta[None])[0] if batched else jacobian(theta)

    monkeypatch.setattr(so3, "SMALL_THETA", 1.0)
    series = run()
    monkeypatch.setattr(so3, "SMALL_THETA", 0.0)
    closed = run()
    assert_allclose(series, closed, rtol=0, atol=1e-13)

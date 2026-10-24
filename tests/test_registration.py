"""Robomath - Test Kabsch and Umeyama on known motions, noise and mirrors."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from test_rotations import TOL

import robomath as rm

RNG = np.random.default_rng(0)
ROTATIONS = Rotation.random(30, rng=1).as_matrix()
TRANSLATIONS = RNG.normal(size=(30, 3))
P = RNG.normal(size=(30, 12, 3))


def move(points, r, t, s=1.0):
    """q_i = s R p_i + t for every set in a batch."""
    return s * np.einsum("bij,bnj->bni", r, points) + t[:, None, :]


def test_kabsch_recovers_a_known_motion():
    """Exact points give back the rotation and translation that made them."""
    r, t = rm.kabsch(P, move(P, ROTATIONS, TRANSLATIONS))
    assert_allclose(r, ROTATIONS, rtol=0, atol=1e-12)
    assert_allclose(t, TRANSLATIONS, rtol=0, atol=1e-12)


def test_kabsch_with_noise_matches_scipy():
    """With noise, the rotation equals scipy's least-squares fit on centred points."""
    q = move(P, ROTATIONS, TRANSLATIONS) + 0.05 * RNG.normal(size=P.shape)
    r, _ = rm.kabsch(P, q)
    for i in range(30):
        pc, qc = P[i] - P[i].mean(0), q[i] - q[i].mean(0)
        expected, _ = Rotation.align_vectors(qc, pc)
        assert_allclose(r[i], expected.as_matrix(), rtol=0, atol=1e-12)


def test_kabsch_equals_nearest_rotation_of_the_cross_covariance():
    """The rotation step is nearest_rotation(H^T), H = sum of p_i q_i^T."""
    q = move(P, ROTATIONS, TRANSLATIONS) + 0.05 * RNG.normal(size=P.shape)
    pc, qc = P - P.mean(1, keepdims=True), q - q.mean(1, keepdims=True)
    h = np.swapaxes(pc, -1, -2) @ qc
    r, _ = rm.kabsch(P, q)
    assert_allclose(r, rm.nearest_rotation(np.swapaxes(h, -1, -2)), rtol=0, atol=1e-12)


def test_kabsch_never_returns_a_mirror():
    """A mirrored copy of the points still gives a proper rotation, det +1."""
    mirrored = P * np.array([1.0, 1.0, -1.0])
    r, _ = rm.kabsch(P, mirrored)
    assert_allclose(np.linalg.det(r), 1.0, rtol=0, atol=1e-12)
    assert_allclose(
        r @ np.swapaxes(r, -1, -2),
        np.broadcast_to(np.eye(3), r.shape),
        rtol=0,
        atol=1e-12,
    )


def test_umeyama_recovers_scale_rotation_translation():
    """Exact points scaled by 0.3 to 3 give back all three."""
    scales = np.linspace(0.3, 3.0, 30)
    s, r, t = rm.umeyama(P, move(P, ROTATIONS, TRANSLATIONS, scales[:, None, None]))
    assert_allclose(s, scales, rtol=0, atol=1e-12)
    assert_allclose(r, ROTATIONS, rtol=0, atol=1e-12)
    assert_allclose(t, TRANSLATIONS, rtol=0, atol=1e-11)


def test_umeyama_with_scale_one_equals_kabsch():
    """With equal-size sets the scale is 1 and the rest matches kabsch."""
    q = move(P, ROTATIONS, TRANSLATIONS)
    s, r, t = rm.umeyama(P, q)
    r2, t2 = rm.kabsch(P, q)
    assert_allclose(s, 1.0, rtol=0, atol=1e-12)
    assert_allclose(r, r2, rtol=0, atol=1e-12)
    assert_allclose(t, t2, rtol=0, atol=1e-12)


def test_single_set_gives_single_motion():
    """One set of points (no batch) gives shapes (3, 3) and (3,)."""
    r, t = rm.kabsch(P[0], move(P, ROTATIONS, TRANSLATIONS)[0])
    assert r.shape == (3, 3)
    assert t.shape == (3,)
    assert_allclose(r, ROTATIONS[0], rtol=0, atol=TOL)

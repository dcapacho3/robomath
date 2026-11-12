"""Report whether the scalar and the batch route give the same bits here.

Every function with a single-input fast path is run both ways - once per item
and once as a batch - on random inputs and the edge angles, and the report says
bit-equal or the largest gap in units in the last place (ulp). The primitive
section then names which building block differs, since a gap always comes from
one of them: the C library behind math.sin against numpy's loops, or the matrix
product. Run it on several machines (the route-report workflow runs it on
Linux, macOS and Windows) and compare the output; one machine alone cannot show
what another will do. The exit code is always 0 - this is a report, not a gate.
"""

import math
import platform

import numpy as np

from robomath import se2, se3, so3
from robomath.euler import euler_zyx_to_matrix, matrix_to_euler_zyx
from robomath.quaternions import quat_multiply, quat_to_axis_angle, slerp

N = 500
RNG = np.random.default_rng(0)


def gap_text(batch, rows):
    """Bit-equal, or the largest spacing-relative gap between the two routes."""
    batch, rows = np.asarray(batch, dtype=float), np.asarray(rows, dtype=float)
    if batch.tobytes() == rows.tobytes():
        return "bit-equal"
    scale = np.maximum(np.abs(batch), np.abs(rows))
    gap = np.abs(batch - rows) / np.spacing(np.where(scale > 0, scale, 1.0))
    return f"max {np.max(gap):.0f} ulp on {np.count_nonzero(gap)} of {gap.size}"


def report(name, fn, batch_args, n=N):
    """One line: the batch result against a loop of single calls."""
    batch = fn(*batch_args)
    rows = np.stack([fn(*(a[i] for a in batch_args)) for i in range(n)])
    print(f"{name:24s} {gap_text(batch, rows)}")


def angles(n):
    """Random angles with the edges 0, pi, under -pi and tiny in front."""
    theta = RNG.uniform(-np.pi, np.pi, n)
    theta[:4] = (0.0, np.pi, np.nextafter(-np.pi, 0.0), 1e-9)
    return theta


def main():
    """Print the route and primitive report for this machine."""
    print(f"{platform.system()} {platform.machine()}, numpy {np.__version__}")

    print("\n-- routes: one batch against a loop of single calls --")
    xyt = RNG.normal(size=(N, 3))
    xyt[:, 2] = angles(N)
    m = se2.xyt_to_se2(xyt)
    m2 = se2.xyt_to_se2(np.roll(xyt, 1, axis=0))
    report("xyt_to_se2", se2.xyt_to_se2, (xyt,))
    report("se2_to_xyt", se2.se2_to_xyt, (m,))
    report("exp_vec_se2", se2.exp_vec_se2, (xyt,))
    report("log_vec_se2", se2.log_vec_se2, (m,))
    report("trans_inv (3x3)", se3.trans_inv, (m,))
    report("rminus_se2", se2.rminus_se2, (m2, m))
    v = RNG.normal(size=(N, 3))
    v[0] = 0.0
    v[1] *= 1e-9 / np.linalg.norm(v[1])
    report("exp_vec (so3)", so3.exp_vec, (v,))
    report("jr", so3.jr, (v,))
    report("jr_inv", so3.jr_inv, (v,))
    r = so3.exp_vec(v)
    report("log_vec (so3)", so3.log_vec, (r,))
    q = np.stack([np.append(a * math.sin(0.3), math.cos(0.3)) for a in v])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q2 = np.roll(q, 1, axis=0)
    report("quat_multiply", quat_multiply, (q, q2))
    report("quat_to_axis_angle", lambda x: quat_to_axis_angle(x)[0], (q,))
    t = RNG.uniform(0, 1, N)
    report("slerp", slerp, (q, q2, t))
    a, b, g = angles(N), angles(N) / 2, angles(N)
    report("euler_zyx_to_matrix", euler_zyx_to_matrix, (a, b, g))
    eul = euler_zyx_to_matrix(a, b, g)
    report(
        "matrix_to_euler_zyx",
        lambda x: np.stack(matrix_to_euler_zyx(x), axis=-1),
        (eul,),
    )

    print("\n-- primitives: where a gap would come from --")
    x = RNG.uniform(-10, 10, 20000)
    y = RNG.uniform(-10, 10, 20000)
    for name, scalar, batch in (
        ("sin", math.sin, np.sin),
        ("cos", math.cos, np.cos),
        ("tan", math.tan, np.tan),
        ("sqrt", math.sqrt, np.sqrt),
    ):
        arg = np.abs(x) if name == "sqrt" else x
        rows = np.array([scalar(float(e)) for e in arg])
        print(f"math.{name} vs np.{name:9s} {gap_text(batch(arg), rows)}")
    rows = np.array([math.atan2(float(s), float(c)) for s, c in zip(x, y)])
    print(f"math.atan2 vs np.arctan2 {gap_text(np.arctan2(x, y), rows)}")
    rows = np.array([math.hypot(float(s), float(c)) for s, c in zip(x, y)])
    print(f"math.hypot vs np.hypot   {gap_text(np.hypot(x, y), rows)}")
    print(f"x**2 vs x*x              {gap_text(x**2, x * x)}")
    mats = se2.xyt_to_se2(RNG.normal(size=(N, 3)))
    rows = np.stack([mats[i] @ mats[i - 1] for i in range(N)])
    print(f"matmul 3x3 vs batched    {gap_text(mats @ np.roll(mats, 1, 0), rows)}")


if __name__ == "__main__":
    main()

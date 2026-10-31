<p align="center">
  <img src="https://raw.githubusercontent.com/dcapacho3/robomath/main/docs/logo.svg" alt="robomath" width="560">
</p>

<p align="center">
  <a href="https://github.com/dcapacho3/robomath/actions/workflows/ci_pytest.yml"><img src="https://github.com/dcapacho3/robomath/actions/workflows/ci_pytest.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/robomath/"><img src="https://img.shields.io/pypi/v/robomath" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/dependencies-NumPy-lightgrey" alt="Depends on NumPy only">
  <img src="https://img.shields.io/badge/coverage-100%25-brightgreen" alt="Coverage 100%">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0 license">
  <a href="https://github.com/dcapacho3/robomath/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-0.1.3-blue" alt="Changelog"></a>
</p>

**The maths of robot pose and motion.**

Robomath is a small Python library for the maths a robot needs to say where it is and
how it moves: rotation matrices, quaternions, Euler angles, and rigid-body transforms
in 2D and 3D, with the Lie group tools (exp, log, plus, minus, Jacobians, adjoints)
that state estimation and optimisation need. The Python package depends on NumPy only.

- **One set of conventions**, written down below and held in every function, so
  results from one part of your code can feed another without a hidden swap.
- **Every function takes batches.** Pass one rotation or ten thousand with the same
  call; there are no loops to write.
- **Checked against other libraries.** Results are compared with `scipy` and
  `modern_robotics`, the laws of each group are tested on generated inputs, and the
  test suite covers every line and branch.
- **Built for loops that run fast.** A single call costs microseconds. Functions do not
  validate their inputs, so validate data once, where it enters your program.

Robomath is young and its API may still change. See [Limits](#limits) for what it does
not do.

## Contents

[Install](#install) · [Quick tour](#quick-tour) · [Conventions](#conventions) ·
[The groups](#the-groups) · [Guide](#guide) · [API reference](#api-reference) ·
[Batches and speed](#batches-and-speed) · [Working with other libraries](#working-with-other-libraries) ·
[Accuracy and testing](#accuracy-and-testing) · [Limits](#limits) ·
[To do](#to-do) · [Changelog](https://github.com/dcapacho3/robomath/blob/main/CHANGELOG.md) · [Learn more](#learn-more) · [Contributing](#contributing) · [License](#license)

## Install

```bash
pip install robomath
```

It needs Python 3.12 or later and NumPy. To work on the code, see
[Contributing](#contributing). Check the install:

```bash
python -c "import robomath as rm; print(rm.rotz(3.14159 / 2).round(2))"
```

## Quick tour

Everything below is a real, tested example. Every line that ends in `# -> value` was
run, and the value is what it prints.

```python
import numpy as np
import robomath as rm

# A rotation about z by 90 degrees sends the x axis to the y axis.
R = rm.rotz(np.pi / 2)
R @ [1, 0, 0]  # -> [0, 1, 0]

# Chain rotations with @. The one next to the vector acts first.
R = rm.rotz(0.3) @ rm.roty(0.2) @ rm.rotx(0.1)

# Rotation vector (axis * angle) <-> matrix. This is the exp and log of the group.
v = rm.log_vec(R)
np.allclose(rm.exp_vec(v), R)  # -> True

# Quaternions are (x, y, z, w). Interpolate along the shortest turn.
q0, q1 = rm.matrix_to_quat(np.eye(3)), rm.matrix_to_quat(rm.rotz(1.0))
rm.quat_to_matrix(rm.slerp(q0, q1, 0.5))[
    :2, :2
]  # -> [[0.8776, -0.4794], [0.4794, 0.8776]]
```

A pose (rotation and translation) has a class, `Transform`, that wraps the functions:

```python
# A tangent vector is a screw motion: turn 90 degrees about z while sliding 1 along x,
# both at once. The slide bends with the turn, so the pose ends at (0.64, 0.64), not (1, 0).
T_ab = rm.Transform.exp([1, 0, 0, 0, 0, np.pi / 2])  # (translation, rotation)
T_ab.translation  # -> [0.6366, 0.6366, 0]

# Move a point given in frame b into frame a.
p_b = np.array([1.0, 0.0, 0.0])
T_ab @ p_b  # -> [0.6366, 1.6366, 0]

# Compose, invert, and take the log back.
T_bc = rm.Transform.from_rt(rm.rotx(0.5), [0, 0, 2])
T_ac = T_ab @ T_bc
(T_ab.inverse() @ T_ac).isclose(T_bc)  # -> True
T_ab.log()  # -> [1, 0, 0, 0, 0, 1.5708]
```

Estimation code compares and corrects poses with plus and minus, which work in the
tangent space (a small vector for a small motion):

```python
T_true = rm.Transform.exp([1.0, 2.0, 0.0, 0.0, 0.0, 0.5])
T_est = T_true.rplus([0.02, -0.01, 0.0, 0.0, 0.0, 0.03])  # a slightly wrong estimate
T_est.rminus(T_true)  # -> [0.02, -0.01, 0, 0, 0, 0.03]
```

Batches use the same functions:

```python
rng = np.random.default_rng(0)
vs = rng.normal(size=(10_000, 3))  # ten thousand rotation vectors
Rs = rm.exp_vec(vs)
Rs.shape  # -> (10000, 3, 3)
rm.log_vec(Rs).shape  # -> (10000, 3)
```

## Conventions

These choices decide whether two libraries agree. Robomath fixes them once.

| Topic | Choice |
|---|---|
| Angles | Radians. |
| Frames | Right-handed. |
| Rotations | **Active**: `R @ v` moves the vector `v` inside one fixed frame. |
| Vectors | Columns, so a product reads right to left. |
| Quaternions | **(x, y, z, w)**, scalar last, Hamilton product. Same as `scipy`, ROS and Eigen's `coeffs()`. Drake, MuJoCo and Ceres put the scalar first; use `quat_xyzw_to_wxyz`. |
| Pose naming | `T_ab` is the pose of frame **b** in frame **a**, so `p_a = T_ab @ p_b`. Inner letters cancel: `T_ac = T_ab @ T_bc`. The inverse swaps them: `T_ba = trans_inv(T_ab)`. A variable may also be spelled `a_from_b`. |
| Tangent vectors | **Translation first, then rotation**: `(rho, theta)`. Extras go last: the log of the scale in Sim(3), the velocity in SE₂(3). SE(2) is `(x, y, theta)`. Modern Robotics, Drake, MuJoCo and GTSAM put rotation first; use `swap_twist`. |
| Plus and minus | **Right** by default: `rplus(M, tau) = M @ Exp(tau)` moves `M` in its own frame. Left versions (`lplus`, `lminus`) act in the world frame. |
| Euler angles | ZYX, intrinsic: `R = rotz(yaw) @ roty(pitch) @ rotx(roll)`. Same as `scipy`'s `"ZYX"`. |
| Axis-angle | Axis unit vector, angle in [0, pi] when read back from a matrix or quaternion. |
| Shapes | Leading dimensions are batch dimensions: a rotation is `(..., 3, 3)`, a vector `(..., 3)`, a quaternion `(..., 4)`, a pose `(..., 4, 4)`. |

Code never reads a tangent vector with literal indices. `twist`, `linear` and
`angular` do it, so the order lives in one place (`LINEAR` and `ANGULAR` in
`robomath/se3.py`).

## The groups

A *group* here is a set of motions you can compose and undo. Each has a matrix form
and a *tangent vector*, the small vector that `exp` turns into a motion and `log`
turns back.

| Group | What it moves | Matrix | Tangent vector | Functions end in |
|---|---|---|---|---|
| SO(2) | a flat turn | 2x2 | `theta` (1) | `so2` |
| SO(3) | a turn in space | 3x3 | rotation vector (3) | `so3`, or none (`hat`, `exp_vec`) |
| SE(2) | a pose on a plane | 3x3 | `(x, y, theta)` (3) | `se2` |
| SE(3) | a pose in space | 4x4 | `(rho, theta)` (6) | `se3` |
| Sim(3) | a pose and a scale | 4x4 | `(rho, theta, sigma)` (7), `scale = exp(sigma)` | `sim3` |
| SE₂(3) | rotation, velocity and position | 5x5 | `(rho, theta, nu)` (9) | `se23` |

Names follow one pattern in every group:

| Name | Meaning |
|---|---|
| `hat_*`, `vee_*` | tangent vector to its matrix, and back |
| `exp_vec_*`, `log_vec_*` | tangent vector to the group element, and back |
| `rplus_*`, `rminus_*`, `lplus_*`, `lminus_*` | move by a tangent vector, and the vector between two elements |
| `adjoint_*` | the matrix that re-expresses a tangent vector in another frame |
| `*_inv` | inverse |

SO(3) has plain names (`hat`, `vee`, `exp_vec`, `log_vec`, `rplus`, `rminus`,
`lplus`, `lminus`) because it is the most used; SE(3)'s adjoint is just `adjoint`.

## Guide

### Rotations and Euler angles

```python
import numpy as np
import robomath as rm

R = rm.rotz(0.3) @ rm.roty(0.2) @ rm.rotx(0.1)
np.allclose(R @ rm.inverse(R), np.eye(3))  # -> True: the inverse is the transpose

# ZYX Euler angles (yaw, pitch, roll), both ways.
R = rm.euler_zyx_to_matrix(0.3, 0.2, 0.1)  # the same R as above
rm.matrix_to_euler_zyx(R)  # -> (0.3, 0.2, 0.1)

# Axis and angle.
axis, angle = rm.matrix_to_axis_angle(rm.rotz(1.0))
axis  # -> [0, 0, 1]
angle  # -> 1.0
rm.axis_angle_to_matrix(axis, angle)  # the same rotation matrix
```

At gimbal lock (pitch = ±90 degrees) the Euler angles are not unique.
`matrix_to_euler_zyx` returns yaw 0 there and puts the rest in roll. Compare
matrices, not angles. For a matrix that has drifted from a true rotation (noise, many
multiplications), `nearest_rotation` returns the closest true rotation:

```python
noisy = rm.rotz(0.5) + 1e-3 * np.random.default_rng(1).normal(size=(3, 3))
fixed = rm.nearest_rotation(noisy)
np.allclose(fixed @ fixed.T, np.eye(3))  # -> True
```

### Quaternions

```python
q = rm.axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), np.pi / 2)  # (x, y, z, w)
q  # -> [0, 0, 0.7071, 0.7071]
rm.quat_to_matrix(q)  # same as rotz(pi / 2)
rm.quat_multiply(q, q)  # rotate by q, then by q again: 180 degrees about z
rm.quat_conjugate(q)  # the inverse of a unit quaternion
rm.quat_to_euler_zyx(q)  # -> (1.5708, 0, 0)
```

`quat_multiply(q1, q2)` rotates by `q2` first, like `R1 @ R2`. `matrix_to_quat` returns
the quaternion with a non-negative scalar, so `q` and `-q` (the same rotation) come
back as one. `quat_normalize` rescales a quaternion that has drifted. `slerp` follows
the shortest arc and switches to a straight blend only when two quaternions nearly
coincide.

### Lie group core on SO(3): exp, log, plus, minus, Jacobians

This part follows Solà, Deray and Atchuthan, *A micro Lie theory for state estimation
in robotics* ([arXiv:1812.01537](https://arxiv.org/abs/1812.01537)). Docstrings cite its
equation numbers.

```python
v = np.array([0.1, -0.2, 0.3])  # a rotation vector: axis * angle
rm.hat(v) @ [1, 0, 0]  # -> the cross product v x [1, 0, 0]
R = rm.exp_vec(v)  # rotation vector -> matrix
rm.log_vec(R)  # -> v

# Plus and minus: move a rotation by a small vector, and recover the vector.
R2 = rm.rplus(R, [0.01, 0.0, 0.0])  # in R's own frame
rm.rminus(R2, R)  # -> [0.01, 0, 0]
rm.lplus([0.01, 0.0, 0.0], R)  # the same step in the world frame

# Jacobians: how a small change of the vector changes the rotation.
rm.jr(v), rm.jl(v)  # right and left Jacobian; jr_inv, jl_inv are the inverses
```

The Jacobians are what an extended Kalman filter on rotations needs to move an
uncertainty from one side of a rotation to the other. The matrix-valued `exp_so3` and
`log_so3` do the same as `exp_vec` and `log_vec` but start from the skew matrix `hat(v)`.

### Poses in 3D: functions and the Transform class

Functions work on plain 4x4 arrays.

```python
R, t = rm.rotz(0.5), np.array([1.0, 2.0, 3.0])
M = rm.rp_to_trans(R, t)  # 4x4 pose [[R, t], [0, 1]]
R_back, t_back = rm.trans_to_rp(M)
rm.trans_inv(M) @ M  # -> the 4x4 identity

tau = rm.twist([1, 0, 0], [0, 0, 0.5])  # (translation, rotation), shape (6,)
rm.linear(tau), rm.angular(tau)  # the two halves
M = rm.exp_vec_se3(tau)  # tangent vector -> pose
rm.log_vec_se3(M)  # -> tau
rm.adjoint(M)  # 6x6: re-expresses a tangent vector in the other frame
```

The adjoint obeys `exp(adjoint(M) @ tau) = M @ exp(tau) @ trans_inv(M)`. It is why a
left step and a right step of the same motion differ: `lplus(adjoint(M) @ tau, M)`
equals `rplus(M, tau)`.

`Transform` holds a pose (or a batch of them) and offers the same operations as
methods:

| Method | Does |
|---|---|
| `Transform(matrix)` | wrap a 4x4 matrix (or `(..., 4, 4)`); checks only the shape |
| `Transform.from_matrix(m)` | same, for data of unknown quality: repairs the rotation and bottom row |
| `Transform.from_rt(r, t)`, `Transform.from_pq(p, q)`, `Transform.exp(tau)`, `Transform.identity()` | other ways to build one |
| `T @ other` | compose with another `Transform` |
| `T @ points` | move points of shape `(..., 3)` |
| `T.inverse()`, `T.log()`, `T.adjoint()` | inverse, tangent vector, 6x6 adjoint |
| `T.rplus(tau)`, `T.lplus(tau)`, `T.rminus(other)`, `T.lminus(other)` | plus and minus |
| `T.interp(other, t)` | the pose a fraction `t` of the way from `T` (0) to `other` (1) |
| `T.rotation`, `T.translation`, `T.matrix`, `T.shape` | parts |
| `T.isclose(other, atol=1e-9)` | compare with a tolerance (`==` is exact) |

A `Transform` is immutable and its matrix is read-only. The constructor does not check
that the rotation is orthogonal (that would slow every `@`); use `from_matrix` for data
from a file, a sensor or a ROS message. `array @ T` raises `TypeError` on purpose;
write `T @ array`.

### Poses in 2D

```python
pose = rm.exp_vec_se2([1.0, 0.0, np.pi / 2])  # 3x3 pose: a screw, as in 3D
rm.log_vec_se2(pose)  # -> [1, 0, 1.5708]

# A robot at (2, 1) facing +y. Drive 0.5 forward in its own frame and turn 0.1.
robot = rm.rp_to_trans(rm.exp_so2(np.pi / 2), [2.0, 1.0])
robot = rm.rplus_se2(robot, [0.5, 0.0, 0.1])
robot[:2, 2]  # -> [1.975, 1.4992]
```

`exp_so2`, `log_so2` and `wrap_angle` handle plain angles, in (-pi, pi]. `rp_to_trans`,
`trans_to_rp` and `trans_inv` accept 3x3 poses as well as 4x4.

Robots report a planar pose as `(x, y, angle)`. `xyt_to_se2` and `se2_to_xyt` convert
it to the matrix and back. These numbers are coordinates, not a tangent vector, so
`xyt_to_se2` is not `exp_vec_se2`; the two agree only at angle 0. `act_se2` moves
points by a pose, and `Transform2D` is the class for it, as `Transform` is for 3D.

```python
robot = rm.Transform2D.from_xyt([2.0, 1.0, np.pi / 2])  # at (2, 1) facing +y
robot @ [1.0, 0.0]  # -> [2, 2]: a point 1 ahead of the robot, in the world frame
robot.inverse().xyt  # -> [-1, 2, -1.5708]: the world origin, seen from the robot
rm.act_se2(rm.xyt_to_se2([1.0, 2.0, np.pi / 2]), [[1.0, 0.0]])  # -> [[1, 3]]
```

`act_se3` and `act_sim3` do the same for 3D poses and similarities.

### Poses with a scale: Sim(3), and fitting points

A similarity moves a point as `p -> s R p + t`. It is the pose group of a camera that
does not know the true size of its world, and the natural result of aligning two point
sets of different size.

```python
rng = np.random.default_rng(2)
p = rng.normal(size=(20, 3))  # points in one frame
R_true = rm.rotz(0.4) @ rm.rotx(-0.3)
q = 1.5 * p @ R_true.T + [0.2, -1.0, 0.5]  # the same points, scaled, turned, moved

R_fit, t_fit = rm.kabsch(p, q / 1.5)  # no scale: rotation and translation only
s, R_fit, t_fit = rm.umeyama(p, q)  # with scale
s  # -> 1.5
M = rm.srt_to_sim3(s, R_fit, t_fit)  # as a 4x4 Sim(3) pose
np.allclose(M[:3, :3] @ p.T + M[:3, 3:], q.T)  # -> True
rm.log_vec_sim3(M)  # (rho, theta, sigma): sigma = log(s)
```

`kabsch(p, q)` finds the rotation and translation that best take points `p` onto the
matching points `q` (least squares). `umeyama` adds the scale. The matches must be
known: point `i` of `p` is point `i` of `q`. Both take batches of point sets of shape
`(..., N, 3)` and always return a true rotation, never a mirror image.

### IMU state: SE₂(3)

The state of a vehicle with an IMU is a rotation `R`, a velocity `v` and a position
`p`, kept together as a 5x5 matrix.

```python
X = rm.rvp_to_se23(rm.rotz(0.2), v=[1.0, 0.0, 0.0], p=[0.0, 0.0, 0.0])
tau = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.02, 0.0])  # (rho, theta, nu)
X2 = rm.rplus_se23(X, tau)
R2, v2, p2 = rm.se23_to_rvp(X2)
rm.rminus_se23(X2, X)  # -> tau
```

It is the state of the invariant EKF and of IMU preintegration (Brossard, Barrau,
Chauchat and Bonnabel, [arXiv:2007.14097](https://arxiv.org/abs/2007.14097)).

### Distance, interpolation and mean

```python
R1, R2 = rm.rotz(0.2), rm.rotz(1.0)
rm.distance_so3(R1, R2)  # -> 0.8  (the angle between them)
rm.interp_so3(R1, R2, 0.5)  # halfway: rotz(0.6)

# Mean of several rotations (or poses): the point that minimises the squared angles.
rotations = rm.exp_vec(np.random.default_rng(3).normal(size=(10, 3)) * 0.2)
rm.mean_so3(rotations)  # 3x3
rm.mean_so3(rotations, weights=np.arange(1.0, 11.0))  # a weighted mean
```

`interp_se3`, `interp_se2`, `mean_se3` and `mean_se2` do the same for poses. There is no
`distance_se3`: a pose mixes metres and radians, so no single length is natural. Use the
two parts of `rminus_se3(M2, M1)`.

The mean starts from the chordal mean (the quaternion-eigenvector method, as in
`scipy`) and refines it, so the answer does not depend on the order of the items. It
is unique when every item lies within 90 degrees of it. Beyond that it warns that the
mean may not be unique; if the loop does not settle it raises `ValueError`.

## API reference

Every function is exported from `robomath`. Docstrings give shapes and the paper
equation where one applies; use `help(rm.name)`.

**Rotations and Euler angles**
`rotx`, `roty`, `rotz`, `inverse` · `euler_zyx_to_matrix`, `matrix_to_euler_zyx`

**Quaternions** (x, y, z, w)
`quat_normalize`, `quat_conjugate`, `quat_multiply`, `axis_angle_to_quat`,
`quat_to_axis_angle`, `quat_to_matrix`, `matrix_to_quat`, `slerp`,
`quat_to_euler_zyx`, `euler_zyx_to_quat`

**SO(2)**
`exp_so2`, `log_so2`, `wrap_angle`

**SO(3)**
`hat`, `vee`, `axis_angle_to_matrix`, `matrix_to_axis_angle`, `exp_vec`, `log_vec`,
`exp_so3`, `log_so3`, `rplus`, `rminus`, `lplus`, `lminus`, `jr`, `jr_inv`, `jl`,
`jl_inv`, `nearest_rotation`

**SE(2)**
`hat_se2`, `vee_se2`, `exp_vec_se2`, `log_vec_se2`, `adjoint_se2`, `rplus_se2`,
`rminus_se2`, `lplus_se2`, `lminus_se2`, `xyt_to_se2`, `se2_to_xyt`, `act_se2`

**SE(3)**
`twist`, `linear`, `angular`, `rp_to_trans`, `trans_to_rp`, `trans_inv`, `hat_se3`,
`vee_se3`, `exp_vec_se3`, `log_vec_se3`, `adjoint`, `rplus_se3`, `rminus_se3`,
`lplus_se3`, `lminus_se3`, `act_se3`

**Sim(3)**
`srt_to_sim3`, `sim3_to_srt`, `sim3_inv`, `hat_sim3`, `vee_sim3`, `exp_vec_sim3`,
`log_vec_sim3`, `adjoint_sim3`, `rplus_sim3`, `rminus_sim3`, `lplus_sim3`, `lminus_sim3`, `act_sim3`

**SE₂(3)**
`rvp_to_se23`, `se23_to_rvp`, `se23_inv`, `hat_se23`, `vee_se23`, `exp_vec_se23`,
`log_vec_se23`, `adjoint_se23`, `rplus_se23`, `rminus_se23`, `lplus_se23`, `lminus_se23`

**Geodesic tools**
`distance_so3`, `interp_so3`, `interp_se2`, `interp_se3`, `mean_so3`, `mean_se2`,
`mean_se3`

**Point-set fitting**
`kabsch`, `umeyama`

**Layout conversion**
`quat_xyzw_to_wxyz`, `quat_wxyz_to_xyzw`, `swap_twist`, `pq_to_trans`, `trans_to_pq`

**Classes**
`Transform`, `Transform2D`

## Batches and speed

Any argument may carry leading dimensions, and they broadcast like NumPy's:

```python
Rs = rm.exp_vec(np.zeros((4, 5, 3)))  # a 4x5 grid of rotations
Rs.shape  # -> (4, 5, 3, 3)

one = rm.exp_vec_se3([0.1, 0, 0, 0, 0, 0.2])  # one pose
taus = np.zeros((100, 6))
rm.rplus_se3(one, taus).shape  # -> (100, 4, 4): one pose against a hundred steps
```

A single call is fast too. Each function has a path for one input that skips the array
work. Measured on one machine, in microseconds per call:

| Call | robomath | other library |
|---|---|---|
| `exp_vec` | 1.1 | 6.1 (`scipy` `Rotation.from_rotvec`) |
| `log_vec` | 1.4 | 28 (`scipy` `Rotation.as_rotvec`) |
| `quat_multiply` | 0.6 | 18 (`scipy` `Rotation` product) |
| `exp_vec_se3` | 5.7 | 27 (`modern_robotics` `MatrixExp6`) |
| `log_vec_se3` | 4.9 | 14 (`modern_robotics` `MatrixLog6`) |

A batch of 1000 is 11 to 75 times faster than a Python loop over the same function
(for example `trans_inv`: 48 µs against 3.6 ms). Your numbers will differ; the ratios
are the point. `exp_vec_sim3` costs about 45 µs per call because it sums a series
instead of using a closed form, and 2 ms for a batch of 1000.

## Working with other libraries

Libraries disagree on the order of a quaternion and of a tangent vector. A mix-up
still runs and gives wrong answers, so robomath names each change of layout:

```python
# scipy stores quaternions as (x, y, z, w): the same order as robomath.
from scipy.spatial.transform import Rotation

R = rm.rotz(0.3)
Rotation.from_quat(rm.matrix_to_quat(R)).as_matrix()  # the same R

# Drake and MuJoCo put the scalar first: (w, x, y, z).
q_wxyz = rm.quat_xyzw_to_wxyz(rm.matrix_to_quat(R))
rm.quat_wxyz_to_xyzw(q_wxyz)  # back to robomath's order

# ROS Pose and TF messages carry a position and an (x, y, z, w) quaternion.
M = rm.pq_to_trans([1.0, 2.0, 3.0], rm.matrix_to_quat(R))
position, quaternion = rm.trans_to_pq(M)

# Modern Robotics orders a twist (rotation, translation). robomath is the reverse.
import modern_robotics as mr

tau = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5])  # (translation, rotation)
mr.MatrixExp6(mr.VecTose3(rm.swap_twist(tau)))  # the same pose as rm.exp_vec_se3(tau)
```

| Order | Libraries |
|---|---|
| Quaternion **x y z w** | robomath, scipy, ROS messages, Eigen `coeffs()`, Pinocchio |
| Quaternion **w x y z** | Drake, MuJoCo, Ceres |
| Tangent vector **translation, rotation** | robomath, Solà's paper, Sophus, manif, Pinocchio |
| Tangent vector **rotation, translation** | Modern Robotics, Drake, MuJoCo, GTSAM Pose3 |

OpenCV and COLMAP store world-to-camera poses. A robomath `T_ab` is the pose of b in
a, so convert with `trans_inv`. ROS message classes are not imported: the functions
take plain arrays, so robomath needs no ROS install.

## Accuracy and testing

Run the suite with `python -m pytest`. It has about 190 distinct tests. Most run on
many inputs each (30 random rotations, every function, several shapes), so pytest
reports 1600+ test cases. A case is one test on one input; the count of different
checks is the smaller number. The suite also requires 100% coverage of every line and
branch. The checks, from the outside in:

- **Other libraries.** Rotations, Euler angles, quaternions, poses and adjoints are
  compared with `scipy` and `modern_robotics` on the same inputs. Sim(3) and SE₂(3)
  are compared with `scipy.linalg.expm` and `logm`.
- **Laws.** Property-based tests ([Hypothesis](https://hypothesis.readthedocs.io))
  generate inputs, including awkward ones, and check that exp and log undo each other,
  `M @ M⁻¹ = I`, the adjoint obeys its definition, and a batch equals a loop of single
  calls.
- **Edge cases.** Zero and tiny angles, angles of exactly π, gimbal lock, and empty
  batches each have tests.
- **Mutation checks.** During development each source file was changed line by line
  (swap a sign, move a limit) to confirm that a test fails.

The tests use a tolerance of 1e-12 for one operation and 1e-9 for chains of several. The Jacobian and Log formulas switch to a series for tiny angles and use
`1 / tan(θ/2)` near π so they stay accurate at both ends.

## Limits

- **Precision.** All results are `float64`. There is no `float32` path and no
  autodiff version.
- **Input checks.** Functions assume valid inputs: unit axes, unit quaternions, true
  rotation matrices. Call `nearest_rotation`, `quat_normalize` or
  `Transform.from_matrix` where data enters. Only `Transform` (wrong shape) and
  `mean_*` (bad weights, no items, no convergence) raise on bad input.
- **Log near π.** At a half turn the axis sign is not defined; `log_vec` returns one
  valid answer. Compare rotations as matrices, not as axes.
- **Euler angles.** Only the ZYX sequence is provided, and gimbal lock loses one angle.
- **The mean.** It is a local answer when the items spread more than about 90 degrees;
  the code warns. It does not search for a global minimum.
- **Registration.** `kabsch` and `umeyama` need known point matches and at least three
  points not on one line. They do not reject outliers.
- **Sim(3) speed.** About 45 µs per single call.
- **Classes.** SE(3) and SE(2) have classes, `Transform` and `Transform2D`. The other
  groups are functions.

## To do

- [ ] **Uncertainty**: moving a covariance through plus, minus and the Jacobians.
  Next release.
- [ ] **Kinematics**: screw axes, product of exponentials, forward and inverse
  kinematics, body and space Jacobians.
- [ ] **Trajectories**: time scaling and smooth paths on SO(3) and SE(3).
- [ ] **More layouts** in `convert`: camera-frame poses, other Euler sequences, SE(2)
  orderings of other libraries.
- [ ] **Classes** for Sim(3) and SE₂(3) next to `Transform` and `Transform2D`.
- [ ] **Single precision** (`float32`) and an autodiff-friendly path.
- [ ] **Other languages**: the same conventions and tests in Rust and in C++. Ideas, not
  started. The conventions above are the contract a port must keep, and this
  library's test inputs can serve as its reference data.

What stays out: filters and estimators, simulators, ROS code, plotting and learned
models. Those belong in the code that uses robomath. The rule is pure array functions,
with no state and no I/O.

## Learn more

- Solà, Deray, Atchuthan, *A micro Lie theory for state estimation in robotics*:
  [arXiv:1812.01537](https://arxiv.org/abs/1812.01537). The SO(3), SE(3), plus and minus
  and Jacobian parts follow it. Joan Solà's talk *Lie theory for the Roboticist* is the
  gentle way in.
- Lynch and Park, [*Modern Robotics*](https://hades.mech.northwestern.edu/index.php/Modern_Robotics),
  chapter 3: rotations, rigid-body motion, twists and the adjoint, free online with
  videos.
- Brossard, Barrau, Chauchat, Bonnabel on SE₂(3):
  [arXiv:2007.14097](https://arxiv.org/abs/2007.14097).
- Kabsch and Umeyama fitting: Nghia Ho's note
  [*Finding optimal rotation and translation between corresponding 3D points*](https://nghiaho.com/?p=671);
  Cyrill Stachniss, [*ICP and point cloud registration, part 1*](https://www.youtube.com/watch?v=dhzLQfDBx2Q).
- Hartley, Trumpf, Dai, Li, *Rotation averaging*, IJCV 2013: the chordal and geodesic means.
- Other libraries with the same ideas: [manif](https://github.com/artivis/manif) (C++),
  [Sophus](https://github.com/strasdat/Sophus) (C++),
  [scipy.spatial.transform](https://docs.scipy.org/doc/scipy/reference/spatial.transform.html),
  [modern_robotics](https://github.com/NxRLab/ModernRobotics).

## Contributing

```bash
git clone https://github.com/dcapacho3/robomath.git
cd robomath
pip install -e . --group dev     # needs pip 25.1 or later
pre-commit install
python -m pytest
```

CI runs `pre-commit` (ruff lint and format) and `pytest --cov-fail-under=100`.
A change should come with tests that keep coverage at 100%. New functions follow the
conventions above, accept leading batch dimensions, and appear in `CASES` in
`tests/test_batching.py`, which checks that a batch equals a loop of single calls.
Compare new maths with an outside library or with the definition (a round trip, an
inverse), not only with itself.

Report bugs and ideas on the [issue tracker](https://github.com/dcapacho3/robomath/issues).

## License

Apache 2.0. Copyright 2026 David Capacho. See [LICENSE](https://github.com/dcapacho3/robomath/blob/main/LICENSE).

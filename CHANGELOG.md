# Changelog

All notable changes to robomath. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- `act_sim3` failed when the pose was a nested list. It now accepts one, as `act_se2`
  and `act_se3` do.

### Changed
- The `act_*` docstrings say that the points need an N axis: one point has shape
  `(1, 2)` or `(1, 3)`.

## [0.1.2] - 2026-10-27

### Added
- `Transform2D`, the SE(2) class, with the same methods as `Transform`, plus `xyt`
  and `from_xyt` for `(x, y, angle)`.
- `xyt_to_se2` and `se2_to_xyt`: a planar pose `(x, y, angle)` to its 3x3 matrix and
  back. These numbers are coordinates, not a tangent vector.
- `act_se2`, `act_se3` and `act_sim3`: move points by a pose or similarity.
- This changelog.

### Fixed
- `wrap_angle` and `log_so2` returned `-pi` for an angle of exactly `-pi`, against
  their documented range `(-pi, pi]`. They now return `pi`. `log_vec_se2` had the
  same fault for a single pose and is fixed too.

## [0.1.1] - 2026-10-26

### Changed
- README updated with the PyPI status.

## [0.1.0] - 2026-10-25

First release on PyPI: SO(2), SO(3), SE(2), SE(3), Sim(3) and SE₂(3), quaternions,
Euler angles, geodesic tools, point-set fitting, layout conversion and `Transform`.

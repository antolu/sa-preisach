from __future__ import annotations

import numpy as np
import pytest

from sa_preisach.utils import (
    CentroidMesh,
    CompositeMesh,
    DefaultMesh,
    DiagonalMesh,
    LineMesh,
    constant_mesh_size,
    create_triangle_mesh,
    default_mesh_size,
    exponential_mesh,
)

X = np.array([0.0, 0.1, 0.5])
Y = np.array([0.2, 0.6, 0.9])
SCALE = 0.1


def test_constant_mesh_size_is_scalar() -> None:
    result = constant_mesh_size(X, Y, SCALE)
    assert float(result) == pytest.approx(SCALE)


def test_default_mesh_size_increases_with_distance() -> None:
    close = default_mesh_size(np.array([0.0]), np.array([0.01]), SCALE)
    far = default_mesh_size(np.array([0.0]), np.array([0.9]), SCALE)
    assert far > close


def test_exponential_mesh_bounded() -> None:
    result = exponential_mesh(X, Y, SCALE)
    assert np.all(result > 0)
    # upper bound is mesh_scale + min_density (default 0.001)
    assert np.all(result <= SCALE + 0.001 + 1e-6)


def test_default_mesh_size_function_matches_functional() -> None:
    fn = DefaultMesh()
    np.testing.assert_allclose(fn(X, Y, SCALE), default_mesh_size(X, Y, SCALE))


def test_default_mesh_size_function_no_init_args() -> None:
    instance = DefaultMesh()
    assert not hasattr(instance, "__dict__") or instance.__dict__ == {}


def test_composite_returns_elementwise_min() -> None:
    fn1 = lambda x, y, s: np.full_like(x, 0.5)  # noqa: E731
    fn2 = lambda x, y, s: np.full_like(x, 0.3)  # noqa: E731
    composite = CompositeMesh(fn1, fn2)
    result = composite(X, Y, SCALE)
    np.testing.assert_allclose(result, 0.3)


def test_composite_min_is_per_element() -> None:
    fn1 = lambda x, y, s: np.array([0.1, 0.8, 0.3])  # noqa: E731
    fn2 = lambda x, y, s: np.array([0.4, 0.2, 0.5])  # noqa: E731
    composite = CompositeMesh(fn1, fn2)
    result = composite(X, Y, SCALE)
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3])


def test_composite_single_fn_passthrough() -> None:
    fn = DefaultMesh()
    composite = CompositeMesh(fn)
    np.testing.assert_allclose(composite(X, Y, SCALE), fn(X, Y, SCALE))


def test_composite_three_fns() -> None:
    fn1 = lambda x, y, s: np.full_like(x, 0.9)  # noqa: E731
    fn2 = lambda x, y, s: np.full_like(x, 0.5)  # noqa: E731
    fn3 = lambda x, y, s: np.full_like(x, 0.2)  # noqa: E731
    composite = CompositeMesh(fn1, fn2, fn3)
    np.testing.assert_allclose(composite(X, Y, SCALE), 0.2)


_N_COORDS = 2
_TOL = 1e-6


def test_create_triangle_mesh_returns_2d_points() -> None:
    pts = create_triangle_mesh(0.2)
    assert pts.ndim == _N_COORDS
    assert pts.shape[1] == _N_COORDS


def test_create_triangle_mesh_points_in_unit_triangle() -> None:
    pts = create_triangle_mesh(0.2)
    beta, alpha = pts[:, 0], pts[:, 1]
    assert np.all(beta >= -_TOL)
    assert np.all(alpha <= 1.0 + _TOL)
    assert np.all(alpha >= beta - _TOL)


def test_diagonal_mesh_fine_at_diagonal() -> None:
    fn = DiagonalMesh(ls=0.05)
    on_diag = fn(np.array([0.3]), np.array([0.3]), SCALE)
    off_diag = fn(np.array([0.0]), np.array([1.0]), SCALE)
    assert on_diag < off_diag


def test_diagonal_mesh_background_is_upper_bound() -> None:
    fn = DiagonalMesh(ls=0.05, background=0.5)
    result = fn(X, Y, SCALE)
    assert np.all(result <= SCALE * 0.5 + 1e-9)


def test_diagonal_mesh_standalone_covers_triangle() -> None:
    fn = DiagonalMesh(ls=0.05, background=1.0)
    far_from_diag = fn(np.array([0.0]), np.array([1.0]), SCALE)
    assert far_from_diag > 0


def test_saturation_corner_fine_at_target() -> None:
    fn = CentroidMesh(ls=0.05)
    at_corner = fn(np.array([0.0]), np.array([1.0]), SCALE)
    far = fn(np.array([0.5]), np.array([0.5]), SCALE)
    assert at_corner < far


def test_saturation_corner_background_is_upper_bound() -> None:
    fn = CentroidMesh(ls=0.05, background=0.5)
    result = fn(X, Y, SCALE)
    assert np.all(result <= SCALE * 0.5 + 1e-9)


def test_saturation_corner_standalone_covers_triangle() -> None:
    fn = CentroidMesh(ls=0.05, background=1.0)
    far_from_corner = fn(np.array([0.5]), np.array([0.5]), SCALE)
    assert far_from_corner > 0


def test_composite_diagonal_and_saturation() -> None:
    diag = DiagonalMesh(ls=0.05, background=1.0)
    sat = CentroidMesh(ls=0.05, background=1.0)
    composite = CompositeMesh(diag, sat)
    # centre of triangle: neither near diagonal nor near saturation corner
    centre_b, centre_a = np.array([0.1]), np.array([0.6])
    # near saturation corner (beta~0, alpha~1)
    corner_b, corner_a = np.array([0.02]), np.array([0.95])
    result_centre = composite(centre_b, centre_a, SCALE)
    result_corner = composite(corner_b, corner_a, SCALE)
    assert result_corner < result_centre


def test_line_mesh_beta_fine_at_value() -> None:
    fn = LineMesh(axis="beta", value=0.0, ls=0.05)
    on_line = fn(np.array([0.0]), np.array([0.5]), SCALE)
    off_line = fn(np.array([0.8]), np.array([0.9]), SCALE)
    assert on_line < off_line


def test_line_mesh_alpha_fine_at_value() -> None:
    fn = LineMesh(axis="alpha", value=0.0, ls=0.05)
    on_line = fn(np.array([0.0]), np.array([0.0]), SCALE)
    off_line = fn(np.array([0.0]), np.array([0.9]), SCALE)
    assert on_line < off_line


def test_line_mesh_background_is_upper_bound() -> None:
    fn = LineMesh(axis="beta", value=0.0, ls=0.05, background=0.5)
    result = fn(X, Y, SCALE)
    assert np.all(result <= SCALE * 0.5 + 1e-9)


def test_line_mesh_min_size_floor() -> None:
    min_size = 0.001
    fn = LineMesh(axis="beta", value=0.0, ls=0.05, min_size=min_size)
    result = fn(np.array([0.0]), np.array([0.5]), SCALE)
    assert np.all(result >= min_size)


def test_line_mesh_invalid_axis_raises() -> None:
    with pytest.raises(ValueError, match="axis"):
        LineMesh(axis="gamma")  # type: ignore[arg-type]


def test_create_triangle_mesh_with_composite_density() -> None:
    fn = CompositeMesh(DefaultMesh(), constant_mesh_size)
    pts = create_triangle_mesh(0.15, mesh_density_function=fn)
    assert pts.shape[1] == _N_COORDS

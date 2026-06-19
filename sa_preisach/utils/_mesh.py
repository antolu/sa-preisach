from __future__ import annotations

import typing

import numpy as np
import pygmsh


def constant_mesh_size(x: np.ndarray, y: np.ndarray, mesh_scale: float) -> np.ndarray:
    return np.array(mesh_scale)


def default_mesh_size(x: np.ndarray, y: np.ndarray, mesh_scale: float) -> np.ndarray:
    return mesh_scale * (0.2 * (np.abs(x - y)) + 0.05)


def exponential_mesh(
    x: np.ndarray,
    y: np.ndarray,
    mesh_scale: float,
    min_density: float = 0.001,
    ls: float = 0.05,
) -> np.ndarray:
    return mesh_scale * (1.0 - np.exp(-np.abs(x - y) / ls)) + min_density


_DENSITY_FUNCTIONS: dict[
    str, typing.Callable[[np.ndarray, np.ndarray, float], np.ndarray]
] = {
    "constant": constant_mesh_size,
    "exponential": exponential_mesh,
    "default": default_mesh_size,
}


def make_mesh_size_function(
    mesh_function: str,
) -> typing.Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Create a mesh size function based on the given string.
    """
    if mesh_function not in _DENSITY_FUNCTIONS:
        msg = f"Invalid mesh size function '{mesh_function}'. Valid options are: {list(_DENSITY_FUNCTIONS.keys())}"
        raise ValueError(msg)

    return _DENSITY_FUNCTIONS[mesh_function]


class DefaultMesh:
    """Linear mesh size that grows with distance from the diagonal.

    Size = mesh_scale * (0.2 * |alpha - beta| + 0.05), giving finer elements
    near the diagonal and coarser away from it.

    Example:
        >>> fn = DefaultMesh()
        >>> pts = create_triangle_mesh(0.1, mesh_density_function=fn)
    """

    def __call__(self, x: np.ndarray, y: np.ndarray, mesh_scale: float) -> np.ndarray:
        return mesh_scale * (0.2 * np.abs(x - y) + 0.05)


class DiagonalMesh:
    """Exponentially fine mesh along the alpha=beta diagonal of the Preisach plane.

    Elements shrink to `min_size` on the diagonal and grow toward
    `mesh_scale * background` away from it. Useful for soft-iron where most
    hysterons switch near zero field (alpha ≈ beta ≈ 0).

    Parameters
    ----------
    ls:
        Length scale controlling how quickly element size grows away from the
        diagonal. Smaller values concentrate refinement in a narrower band.
    background:
        Upper bound on element size as a fraction of `mesh_scale`. Set to 1.0
        to allow full coarsening far from the diagonal; lower values keep the
        rest of the triangle finer.
    min_size:
        Absolute floor on element size. Prevents gmsh from receiving lc=0 on
        the diagonal itself.

    Example:
        >>> fn = DiagonalMesh(ls=0.03, background=0.8)
        >>> pts = create_triangle_mesh(0.1, mesh_density_function=fn)

        Combine with CentroidMesh for soft iron:
        >>> composite = CompositeMesh(
        ...     DiagonalMesh(ls=0.05),
        ...     CentroidMesh(ls=0.05),
        ... )
        >>> pts = create_triangle_mesh(0.1, mesh_density_function=composite)
    """

    def __init__(
        self, ls: float = 0.05, background: float = 1.0, min_size: float = 0.001
    ) -> None:
        self.ls = ls
        self.background = background
        self.min_size = min_size

    def __call__(self, x: np.ndarray, y: np.ndarray, mesh_scale: float) -> np.ndarray:
        fine = mesh_scale * (1.0 - np.exp(-np.abs(x - y) / self.ls)) + self.min_size
        return np.minimum(fine, mesh_scale * self.background)


class LineMesh:
    """Exponentially fine mesh along a horizontal or vertical line on the Preisach plane.

    Elements shrink to `min_size` on the line and grow toward
    `mesh_scale * background` away from it. Useful for refining along
    boundaries such as alpha=0 (no positive switching) or beta=0 (no negative
    switching).

    Parameters
    ----------
    axis:
        Which coordinate to measure distance from. ``"alpha"`` refines along a
        horizontal line at ``value``; ``"beta"`` refines along a vertical line.
    value:
        Position of the line along the chosen axis (default 0.0).
    ls:
        Length scale controlling how quickly element size grows away from the
        line. Smaller values concentrate refinement in a narrower band.
    background:
        Upper bound on element size as a fraction of `mesh_scale`.
    min_size:
        Absolute floor on element size. Prevents gmsh from receiving lc=0 on
        the line itself.

    Example:
        Refine along beta=0 (vertical boundary):
        >>> fn = LineMesh(axis="beta", value=0.0, ls=0.05)
        >>> pts = create_triangle_mesh(0.1, mesh_density_function=fn)

        Combine with DiagonalMesh:
        >>> composite = CompositeMesh(
        ...     DiagonalMesh(ls=0.05),
        ...     LineMesh(axis="alpha", value=0.0, ls=0.05),
        ... )
        >>> pts = create_triangle_mesh(0.1, mesh_density_function=composite)
    """

    def __init__(
        self,
        axis: typing.Literal["alpha", "beta"] = "beta",
        value: float = 0.0,
        ls: float = 0.05,
        background: float = 1.0,
        min_size: float = 0.001,
    ) -> None:
        if axis not in ("alpha", "beta"):
            msg = f"axis must be 'alpha' or 'beta', got '{axis}'"
            raise ValueError(msg)
        self.axis = axis
        self.value = value
        self.ls = ls
        self.background = background
        self.min_size = min_size

    def __call__(self, x: np.ndarray, y: np.ndarray, mesh_scale: float) -> np.ndarray:
        coord = y if self.axis == "alpha" else x
        dist = np.abs(coord - self.value)
        fine = mesh_scale * (1.0 - np.exp(-dist / self.ls)) + self.min_size
        return np.minimum(fine, mesh_scale * self.background)


class CentroidMesh:
    """Exponentially fine mesh centred on an arbitrary point on the Preisach plane.

    Elements shrink to `min_size` at (beta, alpha) and grow toward
    `mesh_scale * background` far from it. Defaults to the saturation corner
    (alpha=1, beta=0) but can be placed anywhere inside the unit triangle.

    Parameters
    ----------
    ls:
        Length scale controlling how quickly element size grows away from the
        target point. Smaller values concentrate refinement in a tighter region.
    background:
        Upper bound on element size as a fraction of `mesh_scale`.
    alpha:
        alpha coordinate (switch-up threshold) of the refinement centre.
    beta:
        beta coordinate (switch-down threshold) of the refinement centre.
    min_size:
        Absolute floor on element size. Prevents gmsh from receiving lc=0 at
        the target point itself.

    Example:
        >>> fn = CentroidMesh(ls=0.08, background=0.6)
        >>> pts = create_triangle_mesh(0.1, mesh_density_function=fn)

        Place the centroid at mid-triangle:
        >>> fn = CentroidMesh(alpha=0.7, beta=0.3)
    """

    def __init__(
        self,
        ls: float = 0.05,
        background: float = 1.0,
        alpha: float = 1.0,
        beta: float = 0.0,
        min_size: float = 0.001,
    ) -> None:
        self.ls = ls
        self.background = background
        self.alpha = alpha
        self.beta = beta
        self.min_size = min_size

    def __call__(self, x: np.ndarray, y: np.ndarray, mesh_scale: float) -> np.ndarray:
        dist = np.sqrt((x - self.beta) ** 2 + (y - self.alpha) ** 2)
        fine = mesh_scale * (1.0 - np.exp(-dist / self.ls)) + self.min_size
        return np.minimum(fine, mesh_scale * self.background)


class CompositeMesh:
    """Combine multiple mesh size functions by taking the elementwise minimum.

    Since gmsh treats the size callback as an upper bound, taking the minimum
    gives the finest resolution requested by any of the component functions at
    each point. Each function is usable standalone (via its `background`
    parameter) and the composite simply tightens wherever any single function
    demands refinement.

    Parameters
    ----------
    *fns:
        Any number of callables with signature (x, y, mesh_scale) -> ndarray.

    Example:
        >>> composite = CompositeMesh(
        ...     DiagonalMesh(ls=0.05),
        ...     CentroidMesh(ls=0.05),
        ... )
        >>> pts = create_triangle_mesh(0.1, mesh_density_function=composite)
    """

    def __init__(
        self, *fns: typing.Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    ) -> None:
        self.fns = fns

    def __call__(self, x: np.ndarray, y: np.ndarray, mesh_scale: float) -> np.ndarray:
        return np.minimum.reduce([fn(x, y, mesh_scale) for fn in self.fns])


def create_triangle_mesh(
    mesh_scale: float,
    mesh_density_function: (
        typing.Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None
    ) = None,
) -> np.ndarray:
    mesh_density_function = mesh_density_function or default_mesh_size
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(
            [
                [0, 0],
                [1, 1],
                [0, 1],
            ],
            mesh_size=0.1,
        )

        # set mesh size with function
        geom.set_mesh_size_callback(
            lambda dim, tag, x, y, z, lc: mesh_density_function(x, y, mesh_scale)
        )

        mesh = geom.generate_mesh()

    return mesh.points[:, :-1]

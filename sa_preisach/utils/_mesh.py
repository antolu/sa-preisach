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


class DefaultMeshSizeFunction:
    def __call__(self, x: np.ndarray, y: np.ndarray, mesh_scale: float) -> np.ndarray:
        return mesh_scale * (0.2 * np.abs(x - y) + 0.05)


class DiagonalMeshSizeFunction:
    def __init__(
        self, ls: float = 0.05, background: float = 1.0, min_size: float = 0.001
    ) -> None:
        self.ls = ls
        self.background = background
        self.min_size = min_size

    def __call__(self, x: np.ndarray, y: np.ndarray, mesh_scale: float) -> np.ndarray:
        fine = mesh_scale * (1.0 - np.exp(-np.abs(x - y) / self.ls)) + self.min_size
        return np.minimum(fine, mesh_scale * self.background)


class SaturationCornerMeshSizeFunction:
    def __init__(
        self,
        ls: float = 0.05,
        background: float = 1.0,
        alpha_target: float = 1.0,
        beta_target: float = 0.0,
        min_size: float = 0.001,
    ) -> None:
        self.ls = ls
        self.background = background
        self.alpha_target = alpha_target
        self.beta_target = beta_target
        self.min_size = min_size

    def __call__(self, x: np.ndarray, y: np.ndarray, mesh_scale: float) -> np.ndarray:
        dist = np.sqrt((x - self.beta_target) ** 2 + (y - self.alpha_target) ** 2)
        fine = mesh_scale * (1.0 - np.exp(-dist / self.ls)) + self.min_size
        return np.minimum(fine, mesh_scale * self.background)


class CompositeMeshSizeFunction:
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

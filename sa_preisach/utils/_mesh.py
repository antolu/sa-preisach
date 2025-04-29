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

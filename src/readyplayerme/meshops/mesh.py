"""Functions to handle mesh data and read it from file."""
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import trimesh

# Abstraction for the mesh type.
Mesh: TypeAlias = trimesh.Trimesh


def read_mesh(filename: str | Path) -> Mesh:
    """Load 3D model data from a file into a Mesh representation.

    :param filename: The path to the to be loaded.
    :return: The loaded mesh object.
    """
    reader = get_mesh_reader(filename)
    return reader(filename)


def get_mesh_reader(filename: str | Path) -> Callable[[str | Path], Mesh]:
    """Return a reader function for a given file extension."""
    if (ext := Path(filename).suffix) in [".glb", ".gltf"]:
        return read_glb
    msg = f"Unsupported file format: {ext}"
    raise NotImplementedError(msg)


def read_glb(filename: str | Path) -> Mesh:
    """Load 3D model data from a GLB file into a Mesh representation.

    :param filename: The path to the GLB file to be loaded.
    :return: The loaded mesh object.
    """
    return trimesh.load(filename, process=False, force="mesh")


def get_edges_from_mesh(mesh: Mesh) -> npt.NDArray[np.int32]:
    """Return the unsorted edges of the mesh.

    :param mesh: The mesh to get the edges from.
    :return: The unsorted edges of the mesh.
    :raises TypeError: If mesh is not a supported mesh type.
    """
    # Check if it's a Trimesh mesh
    if isinstance(mesh, trimesh.Trimesh):
        return mesh.edges

    # TODO: Add additional mesh type checks and return their edges
    """ elif isinstance(mesh, SomeOtherMeshType):
            return mesh.some_other_method_to_get_edges()"""
    msg = f"Unsupported mesh format: {type(mesh)}"
    raise NotImplementedError(msg)


def get_border_vertices(edges: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Return the indices of the vertices on the borders of a mesh.

    :param edges: The unsorted edges of the mesh.
    :return: The indices of the border vertices.
    """
    sorted_edges = np.sort(edges, axis=1)
    unique_edges, edge_triangle_count = np.unique(sorted_edges, return_counts=True, axis=0)
    border_edge_indices = np.where(edge_triangle_count == 1)[0]
    return np.unique(unique_edges[border_edge_indices])

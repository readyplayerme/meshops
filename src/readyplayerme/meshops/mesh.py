"""Functions to handle mesh data and read it from file."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh

from readyplayerme.meshops.types import Edges, Indices, Mesh, VariableLengthArrays, Vertices


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
        return read_gltf
    msg = f"Unsupported file format: {ext}"
    raise NotImplementedError(msg)


def read_gltf(filename: str | Path) -> Mesh:
    """Load 3D model data from a glTF file into a Mesh representation.

    :param filename: The path to the glTF file to be loaded.
    :return: The loaded mesh object.
    """
    return trimesh.load(filename, process=False, force="mesh")


def get_boundary_vertices(edges: Edges) -> Indices:
    """Return the indices of the vertices on the mesh boundary.

    A boundary edge is an edge that only belongs to a single triangle.

    :param edges: The edges of the mesh. Must include all edges by faces (duplicates).
    :return: Vertex indices on mesh boundary.
    """
    sorted_edges = np.sort(edges, axis=1)
    unique_edges, edge_triangle_count = np.unique(sorted_edges, return_counts=True, axis=0)
    border_edge_indices = np.where(edge_triangle_count == 1)[0]
    return np.unique(unique_edges[border_edge_indices])


def get_overlapping_vertices(vertices_pos: Vertices, indices: Indices | None = None) -> VariableLengthArrays:
    """Return the indices of the vertices grouped by the same position.

    Vertices that have the same position belong to a seam.


    :param vertices: All the vertices of the mesh.
    :param indices:  Vertex indices.
    :return: A list of grouped border vertices that share position.
    """
    if indices is None:
        indices = np.arange(len(vertices_pos))

    vertex_positions = vertices_pos[indices]
    rounded_positions = np.round(vertex_positions, decimals=5)
    structured_positions = np.core.records.fromarrays(
        rounded_positions.transpose(), names="x, y, z", formats="f8, f8, f8"
    )
    unique_positions, local_indices = np.unique(structured_positions, return_inverse=True)
    grouped_indices = [
        indices[local_indices == i] for i in range(len(unique_positions)) if (local_indices == i).sum() > 1
    ]

    return grouped_indices

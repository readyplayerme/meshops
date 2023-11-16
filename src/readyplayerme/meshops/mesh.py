"""Functions to handle mesh data and read it from file."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh

from readyplayerme.meshops.types import Edges, Indices, Mesh, TexCoord, UVs


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


def uv_to_texture_space(
    uvs: UVs,
    width: int,
    height: int,
    indices: Indices | None = None,
) -> TexCoord:
    """Convert UV coordinates to texture space coordinates.

    :param uvs: UV coordinates of the mesh.
    :param indices: Indices of the vertices whose UV coordinates are provided.
    :param width: Width of the texture image.
    :param height: Height of the texture image.
    :return: Texture space coordinates as pixel values.
    """
    indices = (
        np.arange(len(uvs)) if indices is None else (np.empty((0, 2), dtype=np.int32) if len(indices) == 0 else indices)
    )

    if np.any(indices >= len(uvs)):
        msg = "Some indices do not have corresponding UV coordinates."
        raise ValueError(msg)

    selected_uvs = uvs[indices]

    if np.any(selected_uvs < 0) or np.any(selected_uvs > 1):
        msg = "UV coordinates are out of bounds (0-1)."
        raise ValueError(msg)

    # Convert UV coordinates to texture space (pixel coordinates)
    # Flip the y-axis as UV (0,0) usually represents bottom-left,
    # while texture space (0,0) is typically top-left.
    texture_space_coords = np.empty((len(selected_uvs), 2), dtype=np.int32)
    if len(selected_uvs) > 0:
        texture_space_coords[:, 0] = (selected_uvs[:, 0] * width).astype(np.int32)
        texture_space_coords[:, 1] = ((1 - selected_uvs[:, 1]) * height).astype(np.int32)

    return texture_space_coords

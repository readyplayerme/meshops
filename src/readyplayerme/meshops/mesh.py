"""Functions to handle mesh data and read it from file."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from readyplayerme.meshops.types import Color, Edges, IndexGroups, Indices, Mesh, Vertices


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


def get_overlapping_vertices(
    vertices_pos: Vertices, indices: Indices | None = None, tolerance: float = 0.00001
) -> IndexGroups:
    """Return the indices of the vertices grouped by the same position.

    :param vertices_pos: All the vertices of the mesh.
    :param indices: Vertex indices.
    :param precision: Tolerance for considering positions as overlapping.
    :return: A list of grouped vertices that share position.
    """
    # Not using try / except because when using an index of -1 gets the last element and creates a false positive
    if indices is None:
        selected_vertices = vertices_pos
    else:
        if len(indices) == 0:
            return []
        if np.any(indices < 0):
            msg = "Negative index value is not allowed."
            raise IndexError(msg)

        if np.max(indices) >= len(vertices_pos):
            msg = "Index is out of bounds."
            raise IndexError(msg)

        selected_vertices = vertices_pos[indices]

    tree = cKDTree(selected_vertices)

    grouped_indices = []
    processed = set()
    for idx, vertex in enumerate(selected_vertices):
        if idx not in processed:
            # Find all points within the tolerance distance
            neighbors = tree.query_ball_point(vertex, tolerance)
            if len(neighbors) > 1:  # Include only groups with multiple vertices
                # Translate to original indices if needed
                group = np.array(neighbors, dtype=np.uint32) if indices is None else indices[neighbors]
                grouped_indices.append(group)
            # Mark these points as processed
            processed.update(neighbors)

    return grouped_indices


def blend_colors(colors: Color, index_groups: IndexGroups) -> Color:
    """Blends colors according to the given groups.

    :param index_groups: A list of groups of indices.
    :param vertex_colors: A list of colors .
    :return: Array containing the blended colors .
    """
    if not isinstance(colors, np.ndarray):
        msg = "colors must be a NumPy array"
        raise ValueError(msg)
    if not len(colors):
        return np.array(colors)

    blended_colors = np.copy(colors)
    if not index_groups or index_groups == []:
        return blended_colors

    try:
        colors[np.hstack(index_groups)]
    except IndexError as error:
        msg = f"Index out of bounds in index groups: {error}"

        raise IndexError(msg) from error

    # Blending process
    for group in index_groups:
        if len(group):
            blended_colors[group] = np.mean(colors[group], axis=0)

    return blended_colors

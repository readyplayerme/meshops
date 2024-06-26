"""Functions to handle mesh data and read it from file."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from readyplayerme.meshops.material import Material
from readyplayerme.meshops.types import Edges, Faces, IndexGroups, Indices, Normals, PixelCoord, UVs, Vertices


@dataclass()
class Mesh:
    """Mesh data type.

    This class serves as an abstraction for loading mesh data from different file formats.
    """

    vertices: Vertices
    edges: Edges
    faces: Faces
    uv_coords: UVs | None = None
    normals: Normals | None = None
    material: Material | None = None


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
    try:
        loaded = trimesh.load(filename, process=False, force="mesh")
        assert isinstance(loaded, trimesh.Trimesh), "Loaded object is not a Trimesh."  # noqa: S101  # For type checker.
    except ValueError as error:
        msg = f"Error loading {filename}: {error}"
        raise OSError(msg) from error
    # Convert the loaded trimesh into a Mesh object for abstraction.
    try:
        # trimesh doesn't load UVs if there's no material.
        uvs = loaded.visual.uv  # Fails if it has ColorVisuals instead of TextureVisuals.
    except AttributeError:
        if isinstance(loaded.visual, trimesh.visual.color.ColorVisuals):
            uvs = loaded.visual.to_texture().uv  # Sets the shape of UVs, but values are all 0.5.
        else:
            uvs = None
    normals = loaded.vertex_normals
    try:
        material = Material.from_trimesh_material(loaded.visual.material)
    except AttributeError:
        material = None
    return Mesh(
        vertices=np.array(loaded.vertices),
        faces=np.array(loaded.faces),
        edges=loaded.edges,
        uv_coords=uvs,
        normals=normals,
        material=material,
    )


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
    vertices_pos: Vertices, indices: Indices | None = None, tolerance: float = 1e-5
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


def faces_to_edges(faces: Faces) -> Edges:
    """Return edges of the faces.

    :param faces: Faces to convert to edge representation.
    :return: Edges contained in the faces.
    """
    # Split faces into first 2 columns, last 2 columns and first + last column.
    return np.hstack((faces[:, :2], faces[:, 1:], np.roll(faces, 1, axis=1)[:, :2])).reshape(faces.size, 2)


def uv_to_image_coords(
    uvs: UVs,
    width: int,
    height: int,
    indices: Indices | None = None,
) -> PixelCoord:
    """Convert UV coordinates to image space coordinates.

    :param uvs: UV coordinates.
    :param indices: Optional subset of UV indices for which to retrieve pixel coordinates.
    :param width: Width of the image in pixels.
    :param height: Height of the image in pixels.
    :return: Coordinates in image space given the input width and height.
    """
    try:
        selected_uvs = uvs if indices is None else uvs[indices]
    except IndexError as error:
        msg = f"Index {np.where(indices>=len(uvs))[0]} is out of bounds for UVs with shape {uvs.shape}."  # type: ignore
        raise IndexError(msg) from error
    if not len(selected_uvs):
        return np.empty((0, 2), dtype=np.uint16)

    # Wrap UV coordinates within the range [0, 1].
    wrapped_uvs = np.mod(selected_uvs, 1)

    # With wrapping, we keep the max 1 as 1 and not transpose into the next space.
    wrapped_uvs[selected_uvs == 1] = 1

    # Convert UV coordinates to texture space (pixel coordinates)
    img_coords = np.empty((len(selected_uvs), 2), dtype=np.uint16)
    img_coords[:, 0] = (wrapped_uvs[:, 0] * (width - 0.5)).astype(np.uint16)
    img_coords[:, 1] = ((1 - wrapped_uvs[:, 1]) * (height - 0.5)).astype(np.uint16)

    return img_coords


def get_faces_image_coords(
    faces: Faces,
    uvs: UVs,
    width: int,
    height: int,
) -> PixelCoord:  # Shape (n_faces, 3, 2)
    """Return the pixel coordinates of the vertices of each face.

    :param faces: The faces of the mesh.
    :param uvs: The UV coordinates for the vertices of the faces.
    :param width: Width of the image in pixels.
    :param height: Height of the image in pixels.
    :return: Image space coordinates of faces as shape (n_faces, 3, 2).
    """
    vert_indices = faces.ravel()
    try:
        uv_coords = uvs[vert_indices]
    except IndexError as error:
        msg = f"Vertex index {np.where(vert_indices>=len(uvs))[0]} is out of bounds for UVs with shape {uvs.shape}."
        raise IndexError(msg) from error
    pixel_coords = uv_to_image_coords(uv_coords, width, height)
    # Reshape the pixel coordinates to match the faces.
    return pixel_coords.reshape((*faces.shape, 2))

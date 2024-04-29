"""Unit tests for the mesh module."""

import types
from collections.abc import Callable
from pathlib import Path
from typing import Union, get_args, get_origin

import numpy as np
import pytest

from readyplayerme.meshops import mesh
from readyplayerme.meshops.types import Faces, IndexGroups, Indices, PixelCoord, UVs, Vertices


class TestReadMesh:
    """Test suite for the mesh reader functions."""

    @pytest.mark.parametrize(
        "filepath, expected",
        [("test.glb", mesh.read_gltf), (Path("test.glb"), mesh.read_gltf), ("test.gltf", mesh.read_gltf)],
        ids=["str .glb", "Path .glb", "str .gltf"],
    )
    def test_get_mesh_reader_glb(self, filepath: str | Path, expected: Callable[[str | Path], mesh.Mesh]):
        """Test the get_mesh_reader function with a .glb file path."""
        reader = mesh.get_mesh_reader(filepath)
        assert callable(reader), "The mesh reader should be a callable function."
        assert reader == expected, "The reader function for .glTF files should be read_gltf."

    @pytest.mark.parametrize(
        "filepath",
        ["test", "test.obj", Path("test.stl"), "test.fbx", "test.abc", "test.ply"],
        ids=["no extension", ".obj", ".stl", ".fbx", ".abc", ".ply"],
    )
    def test_get_mesh_reader_should_fail(self, filepath: str | Path):
        """Test the get_mesh_reader function with an unsupported file format."""
        with pytest.raises(NotImplementedError):
            mesh.get_mesh_reader(filepath)

    def test_read_mesh_gltf(self, gltf_simple_file: str | Path):
        """Test the read_gltf function returns the expected type."""
        result = mesh.read_gltf(gltf_simple_file)
        # Check the result has all the expected attributes and of the correct type.
        for attr in mesh.Mesh.__annotations__:
            assert hasattr(result, attr), f"The mesh class should have a '{attr}' attribute."
            # Find the type so we can use it as a second argument to isinstance.
            tp = get_origin(mesh.Mesh.__annotations__[attr])
            if tp is Union or tp is types.UnionType:
                tp = get_args(mesh.Mesh.__annotations__[attr])
                # Loop through the types in the union and check if the result is compatible with any of them.
                matched = False
                for type_ in tp:
                    # Check original type definition. isinstance doesn't work with None, but with NoneType.
                    origin = o if (o := get_origin(type_)) is not None else type_
                    matched |= isinstance(getattr(result, attr), origin)
                assert matched, f"The '{attr}' attribute should be compatible with {tp}."
            else:
                assert isinstance(
                    getattr(result, attr), tp
                ), f"The '{attr}' attribute should be compatible with {mesh.Mesh.__annotations__[attr]}."


def test_get_boundary_vertices(mock_mesh: mesh.Mesh):
    """Test the get_boundary_vertices function returns the expected indices."""
    boundary_vertices = mesh.get_boundary_vertices(mock_mesh.edges)

    assert np.array_equiv(
        np.sort(boundary_vertices), [0, 2, 4, 6, 7, 9, 10]
    ), "The vertices returned by get_border_vertices do not match the expected vertices."


@pytest.mark.parametrize(
    "vertices, indices, precision, expected",
    [
        # Vertices from our mock mesh.
        ("mock_mesh", np.array([0, 2, 4, 6, 7, 9, 10]), 0.1, [np.array([9, 10])]),
        # Close positions, but with imprecision.
        (
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [0.99998, 0.99998, 0.99998],
                    [0.49998, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.50001, 0.50001, 0.50001],
                ]
            ),
            np.array([0, 1, 2, 3, 4]),
            0.0001,
            [np.array([0, 1]), np.array([2, 3, 4])],
        ),
        # Overlapping vertices, None indices given.
        (
            np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            None,
            0.1,
            [np.array([0, 1])],
        ),
        # Overlapping vertices, but empty indices given.
        (np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), np.array([], dtype=np.int32), 0.1, []),
    ],
    ids=[
        "mock_mesh",
        "close positions with imprecision",
        "overlapping vertices, None indices",
        "overlapping vertices, empty indices",
    ],
)
def test_get_overlapping_vertices(
    vertices: Vertices, indices: Indices, precision: float, expected: IndexGroups, request: pytest.FixtureRequest
):
    """Test the get_overlapping_vertices functions returns the expected indices groups."""
    # Get vertices from the fixture if one is given.
    if isinstance(vertices, str) and vertices == "mock_mesh":
        vertices = request.getfixturevalue("mock_mesh").vertices

    grouped_vertices = mesh.get_overlapping_vertices(vertices, indices, precision)

    assert len(grouped_vertices) == len(expected), "Number of groups doesn't match expected"
    for group, exp_group in zip(grouped_vertices, expected, strict=False):
        np.testing.assert_array_equal(group, exp_group, f"Grouped vertices {group} do not match expected {exp_group}")


@pytest.mark.parametrize(
    "indices",
    [
        # Case with index out of bounds (higher than max)
        np.array([0, 3], dtype=np.uint16),
        # Case with index out of bounds (negative index)
        np.array([0, -1], dtype=np.int32),  # Using int32 to allow negative values
    ],
    ids=["index out of bounds (>max)", "index out of bounds (<min)"],
)
def test_get_overlapping_vertices_should_fail(indices):
    """Test that get_overlapping_vertices function raises an exception for out of bounds indices."""
    vertices = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    with pytest.raises(IndexError):
        mesh.get_overlapping_vertices(vertices, indices)


def test_faces_to_edges(mock_mesh: mesh.Mesh):
    """Test the faces_to_edges function returns the expected edges."""
    edges = mesh.faces_to_edges(mock_mesh.faces)

    np.testing.assert_array_equal(
        edges, mock_mesh.edges, "The edges returned by faces_to_edges do not match the expected edges."
    )


@pytest.mark.parametrize(
    "uvs, width, height, indices, expected",
    [
        # Simple UV conversion with specific indices
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 100, 100, np.array([0]), np.array([[49, 49]])),
        # Full range UV conversion without specific indices
        (np.array([[0.0, 0.0], [1.0, 1.0]]), 200, 200, None, np.array([[0, 199], [199, 0]])),
        # Near 0 and 1 values
        (
            np.array([[0.0001, 0.9999], [0.9999, 0.0001]]),
            200,
            200,
            np.array([0, 1]),
            np.array([[0, 0], [199, 199]]),
        ),
        # Empty indices
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 50, 50, np.array([], dtype=np.uint8), np.empty((0, 2))),  # FixMe
        # UV coordinates out of range  (non square tex - negative values)
        (np.array([[-0.5, 1.5], [1.0, -1.0]]), 10, 100, np.array([0, 1]), np.array([[4, 49], [9, 99]])),
        # UV coordinates out of range (wrapped - negative values)
        (np.array([[-0.25, 1.5], [-2.0, -1.0]]), 100, 100, np.array([0, 1]), np.array([[74, 49], [0, 99]])),
        # UV coordinates for non square
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 124, 10024, np.array([0]), np.array([[61, 5011]])),
        # 1px image
        (np.array([[0.5, 0.5], [-1, 1], [0, 0]]), 1, 1, np.array([0, 1, 2]), np.array([[0, 0], [0, 0], [0, 0]])),
        # 0 px image
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 0, 0, np.array([0]), np.array([[0, 0]])),
    ],
    ids=[
        "simple UV conversion",
        "full range UV conversion",
        "near 0 and 1 values",
        "empty indices",
        "out of range UV",
        "wrapped UV",
        "non square UV",
        "1px image",
        "0px image",
    ],
)
def test_uv_to_image_coords(uvs: UVs, width: int, height: int, indices: Indices, expected: PixelCoord):
    """Test the uv_to_texture_space function returns the correct texture space coordinates."""
    image_space_coords = mesh.uv_to_image_coords(uvs, width, height, indices)
    np.testing.assert_array_equal(image_space_coords, expected, "Image space coordinates do not match expected values.")


@pytest.mark.parametrize(
    "uvs, width, height, indices",
    [
        # Too many indices
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 100, 100, np.array([0, 1, 2])),
        # No UV coord
        (np.array([]), 1, 1, np.array([0, 1, 2])),
    ],
    ids=["too many indices", "no UV coord"],
)
def test_uv_to_image_coords_should_fail(uvs: UVs, width: int, height: int, indices: Indices):
    """Test the uv_to_image_space function raises expected exceptions."""
    with pytest.raises(IndexError):
        mesh.uv_to_image_coords(uvs, width, height, indices)


def test_get_faces_image_coords(mock_mesh):
    """Test the get_faces_image_coords function with valid inputs."""
    output = mesh.get_faces_image_coords(mock_mesh.faces, mock_mesh.uv_coords, 8, 8)
    expected = np.array(
        [
            [[5, 1], [0, 1], [3, 3]],
            [[3, 3], [4, 3], [5, 1]],
            [[4, 3], [4, 5], [6, 3]],
            [[4, 5], [3, 4], [2, 4]],
            [[3, 3], [3, 4], [4, 3]],
            [[2, 4], [0, 1], [0, 6]],
            [[3, 3], [2, 4], [3, 4]],
            [[5, 1], [5, 0], [0, 1]],
            [[2, 4], [3, 3], [0, 1]],
            [[4, 3], [3, 4], [4, 5]],
            [[4, 5], [0, 6], [4, 7]],
            [[6, 3], [4, 5], [4, 7]],
            [[4, 5], [2, 4], [0, 6]],
        ],
    )
    assert output.shape == (len(mock_mesh.faces), 3, 2), "The image coordinates' shape should be (n_faces, 3, 2)."
    np.testing.assert_array_equal(output, expected, "The image coordinates should match the expected coordinates.")


@pytest.mark.parametrize(
    "faces, uvs, width, height",
    [
        # Too few UV coords
        (np.array([[0, 1, 2]]), np.array([[0.5, 0.5], [0.25, 0.75]]), 100, 100),
        # No UV coords
        (np.array([[0, 1, 2]]), np.array([]), 1, 1),
        # No faces
        (np.array([[]]), np.array([[0.5, 0.5], [0.25, 0.75], [0.0, 0.0]]), 1, 1),
    ],
    ids=["too few UV coords", "no UV coords", "no faces"],
)
def test_get_faces_image_coords_should_fail(faces: Faces, uvs: UVs, width: int, height: int):
    """Test the get_faces_image_coords function raises expected exceptions."""
    with pytest.raises(IndexError):
        mesh.get_faces_image_coords(faces, uvs, width, height)

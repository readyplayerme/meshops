"""Unit tests for the mesh module."""
import types
from collections.abc import Callable
from pathlib import Path
from typing import Union, get_args, get_origin

import numpy as np
import pytest

from readyplayerme.meshops import mesh
from readyplayerme.meshops.types import IndexGroups, Indices, Mesh, PixelCoord, UVs, Vertices


class TestReadMesh:
    """Test suite for the mesh reader functions."""

    @pytest.mark.parametrize(
        "filepath, expected",
        [("test.glb", mesh.read_gltf), (Path("test.glb"), mesh.read_gltf), ("test.gltf", mesh.read_gltf)],
    )
    def test_get_mesh_reader_glb(self, filepath: str | Path, expected: Callable[[str | Path], Mesh]):
        """Test the get_mesh_reader function with a .glb file path."""
        reader = mesh.get_mesh_reader(filepath)
        assert callable(reader), "The mesh reader should be a callable function."
        assert reader == expected, "The reader function for .glTF files should be read_gltf."

    @pytest.mark.parametrize("filepath", ["test", "test.obj", Path("test.stl"), "test.fbx", "test.abc", "test.ply"])
    def test_get_mesh_reader_unsupported(self, filepath: str | Path):
        """Test the get_mesh_reader function with an unsupported file format."""
        with pytest.raises(NotImplementedError):
            mesh.get_mesh_reader(filepath)

    def test_read_mesh_gltf(self, gltf_simple_file: str | Path):
        """Test the read_gltf function returns the expected type."""
        result = mesh.read_gltf(gltf_simple_file)
        # Check the result has all the expected attributes and of the correct type.
        for attr in Mesh.__annotations__:
            assert hasattr(result, attr), f"The mesh class should have a '{attr}' attribute."
            # Find the type so we can use it as a second argument to isinstance.
            tp = get_origin(Mesh.__annotations__[attr])
            if tp is Union or tp is types.UnionType:
                tp = get_args(Mesh.__annotations__[attr])
                # Loop through the types in the union and check if the result is compatible with any of them.
                assert any(
                    isinstance(getattr(result, attr), get_origin(t)) for t in tp
                ), f"The '{attr}' attribute should be compatible with {tp}."
            else:
                assert isinstance(
                    getattr(result, attr), tp
                ), f"The '{attr}' attribute should be compatible with {Mesh.__annotations__[attr]}."


def test_get_boundary_vertices(mock_mesh: Mesh):
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
        assert np.array_equal(group, exp_group), f"Grouped vertices {group} do not match expected {exp_group}"


@pytest.mark.parametrize(
    "indices",
    [
        # Case with index out of bounds (higher than max)
        np.array([0, 3], dtype=np.uint16),
        # Case with index out of bounds (negative index)
        np.array([0, -1], dtype=np.int32),  # Using int32 to allow negative values
    ],
)
def test_get_overlapping_vertices_error_handling(indices):
    """Test that get_overlapping_vertices function raises an exception for out of bounds indices."""
    vertices = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    with pytest.raises(IndexError):
        mesh.get_overlapping_vertices(vertices, indices)


@pytest.mark.parametrize(
    "uvs, width, height, indices, expected",
    [
        # Simple UV conversion with specific indices
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 100, 100, np.array([0]), np.array([[49, 49]])),
        # Full range UV conversion without specific indices
        (np.array([[0, 0], [1, 1]]), 200, 200, None, np.array([[0, 199], [199, 0]])),
        # Near 0 and 1 values
        (np.array([[0.0001, 0.9999], [0.9999, 0.001]]), 200, 200, np.array([0, 1]), np.array([[0, 0], [199, 199]])),
        # Empty indices
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 50, 50, np.array([]), np.empty((0, 2))),
        # UV coordinates out of range  (non square tex - negative values)
        (np.array([[-0.5, 1.5], [1, -1]]), 10, 100, np.array([0, 1]), np.array([[4, 49], [9, 99]])),
        # UV coordinates out of range (wrapped - negative values)
        (np.array([[-0.25, 1.5], [-2, -1]]), 100, 100, np.array([0, 1]), np.array([[74, 49], [0, 99]])),
        # UV coordinates for non square
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 124, 10024, np.array([0]), np.array([[61, 5011]])),
    ],
)
def test_uv_to_texture_space(uvs: UVs, width: int, height: int, indices: Indices, expected: PixelCoord):
    """Test the uv_to_texture_space function returns the correct texture space coordinates."""
    texture_space_coords = mesh.uv_to_texture_space(uvs, width, height, indices)
    assert np.array_equal(texture_space_coords, expected), "Texture space coordinates do not match expected values."


@pytest.mark.parametrize(
    "uvs, width, height, indices",
    [
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 100, 100, np.array([0, 1, 2])),
    ],
)
def test_uv_to_texture_space_exceptions(uvs: UVs, width: int, height: int, indices: Indices):
    """Test the uv_to_texture_space function raises expected exceptions."""
    with pytest.raises(IndexError):
        mesh.uv_to_texture_space(uvs, width, height, indices)

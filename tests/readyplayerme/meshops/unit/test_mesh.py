"""Unit tests for the mesh module."""
import types
from collections.abc import Callable
from pathlib import Path
from typing import Union, get_args, get_origin

import numpy as np
import pytest

from readyplayerme.meshops import mesh
from readyplayerme.meshops.types import Indices, Mesh, TexCoord, UVs


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
    "uvs, width, height, indices, expected,exception",
    [
        # Test case 1: simple UV conversion with specific indices
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 100, 100, np.array([0]), np.array([[50, 50]]), None),
        # Test case 2: full range UV conversion without specific indices
        (np.array([[0, 0], [1, 1]]), 200, 200, None, np.array([[0, 200], [200, 0]]), None),
        # Test case 3: empty indices
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 50, 50, np.array([]), np.empty((0, 2)), None),
        # Test case 4: UV coordinates out of range
        (np.array([[-0.5, 1.5], [2, -1]]), 100, 100, np.array([0, 1]), None, ValueError),
        # Test case 5: Indices without corresponding UVs
        (np.array([[0.5, 0.5], [0.25, 0.75]]), 100, 100, np.array([0, 1, 2]), None, ValueError),
    ],
)
def test_uv_to_texture_space(
    uvs: UVs,
    width: int,
    height: int,
    indices: Indices,
    expected: TexCoord,
    exception: type[BaseException] | tuple[type[BaseException], ...],
):
    """Test the uv_to_texture_space function returns the correct texture space coordinates."""
    if exception:
        with pytest.raises(exception):
            mesh.uv_to_texture_space(uvs, width, height, indices)
    else:
        texture_space_coords = mesh.uv_to_texture_space(uvs, width, height, indices)
        assert np.array_equal(texture_space_coords, expected), "Texture space coordinates do not match expected values."
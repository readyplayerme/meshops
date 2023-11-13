"""Unit tests for the mesh module."""
import sys
from pathlib import Path

import numpy as np

from readyplayerme.meshops import mesh

mocks_path = Path(__file__).parent.parent.parent / "mocks"

# Add the project root to the Python path to import mock_data.py
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def test_read_glb():
    """Test the read_glb function returns the expected type."""
    filename = mocks_path / "input-mesh.glb"
    result = mesh.read_glb(filename)
    assert isinstance(result, mesh.Mesh), "The result should be an instance of mesh.Mesh"


def test_edges_from_mesh_trimesh():
    """Test the get_edges_from_mesh function returns expected edges."""
    from tests.mocks import mock_data

    expected_edges = mock_data.EDGES
    edges = mesh.get_edges_from_mesh(mock_data.mock_mesh_trimesh)

    assert np.array_equal(
        edges, expected_edges
    ), "The edges returned by get_edges_from_mesh do not match the expected edges."


def test_border_vertices():
    """Test the get_border_vertices function returns the expected indices."""
    from tests.mocks import mock_data

    edges = mock_data.EDGES
    border_vertices = mesh.get_border_vertices(edges)

    assert np.array_equiv(
        np.sort(border_vertices), [0, 2, 4, 6, 7, 9, 10]
    ), "The vertices returned by get_border_vertices do not match the expected vertices."


def test_glb_border_vertices_flow():
    """Test the flow of extracting border vertices from a glb file."""
    filename = mocks_path / "uv-seams.glb"
    local_mesh = mesh.read_glb(filename)
    edges = mesh.get_edges_from_mesh(local_mesh)
    border_vertices = mesh.get_border_vertices(edges)
    assert np.array_equiv(
        np.sort(border_vertices), [0, 2, 4, 6, 7, 9, 10]
    ), "The vertices returned by get_border_vertices do not match the expected vertices."

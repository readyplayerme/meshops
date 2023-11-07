"""Unit tests for the mesh module."""
from pathlib import Path

from readyplayerme.texturesynthesis import mesh

mocks_path = Path(__file__).parent.parent.parent / "mocks"


def test_read_glb():
    """Test the read_glb function returns the expected type."""
    with mocks_path / "input-mesh.glb" as filename:
        result = mesh.read_glb(filename)
    assert isinstance(result, mesh.Mesh), "The result should be an instance of mesh.Mesh"


def test_border_vertices():
    """Test the get_border_vertices function returns the expected indices."""
    import numpy as np

    # TODO: create a fixture for loading the file, or mock it, loading the mesh should not be part of the test!
    with mocks_path / "uv-seams.glb" as filename:
        geo = mesh.read_glb(filename)
    border_vertices = mesh.get_border_vertices(geo)

    assert np.array_equiv(np.sort(border_vertices), [0, 2, 4, 6, 7, 9, 10])

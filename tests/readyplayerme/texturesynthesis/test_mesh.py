"""Unit tests for the mesh module."""
from pathlib import Path

import trimesh

from readyplayerme.texturesynthesis import mesh

mocks_path = Path(__file__).parent.parent.parent / "mocks"


def test_read_glb():
    """Test the read_glb function returns the expected type."""
    # Arrange
    with mocks_path / "input_mesh.glb" as filename:
        # Act
        result = mesh.read_glb(filename)

        # Assert
        assert isinstance(result, trimesh.Trimesh), "The result should be an instance of trimesh.Trimesh"

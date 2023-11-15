"""Pytest fixtures for meshops tests."""
import pytest


@pytest.fixture
def gltf_simple_file():
    """Return a path to a simple glTF file."""
    from importlib.resources import files

    import tests.mocks

    return files(tests.mocks).joinpath("uv-seams.glb")

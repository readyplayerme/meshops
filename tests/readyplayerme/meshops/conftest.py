"""Pytest fixtures for meshops tests."""
import pytest


@pytest.fixture
def gltf_simple_file():
    """Return a path to a simple glTF file."""
    from importlib.resources import files

    import tests.mocks

    return files(tests.mocks).joinpath("uv-seams.glb")


@pytest.fixture
def gltf_file_with_diffuse():
    """Return a path to a glTF file with a diffuse texture in it."""
    from importlib.resources import files

    import tests.mocks

    return files(tests.mocks).joinpath("input-mesh.glb")


@pytest.fixture
def load_local_image():
    """Return a path to an image file."""
    from importlib.resources import files

    import tests.mocks

    return files(tests.mocks).joinpath("input-img.png")

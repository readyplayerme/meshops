"""Pytest fixtures for meshops tests."""
import pytest
import skimage as ski


@pytest.fixture
def gltf_simple_file():
    """Return a path to a simple glTF file."""
    from importlib.resources import files

    import tests.mocks

    return files(tests.mocks).joinpath("uv-seams.glb")


@pytest.fixture
def gltf_file_no_material():
    """Return a path to a glTF file that does not have a material."""
    from importlib.resources import files

    import tests.mocks

    return files(tests.mocks).joinpath("no-material.glb")


@pytest.fixture
def gltf_file_with_basecolor_texture():
    """Return a path to a glTF file that contains a baseColorTexture."""
    from importlib.resources import files

    import tests.mocks

    return files(tests.mocks).joinpath("input-mesh.glb")


@pytest.fixture
def mock_image():
    """Return an image as a numpy array."""
    from importlib.resources import files

    import tests.mocks

    filepath = files(tests.mocks).joinpath("input-img.png")
    return ski.io.imread(filepath)


@pytest.fixture
def mock_image_blended():
    """Return an image as a numpy array."""
    from importlib.resources import files

    import tests.mocks

    filepath = files(tests.mocks).joinpath("input-img-blended.png")
    return ski.io.imread(filepath)

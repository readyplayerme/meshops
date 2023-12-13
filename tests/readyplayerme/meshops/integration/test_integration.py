from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import readyplayerme.meshops.mesh as mops


def test_boundary_vertices_from_file(gltf_simple_file: str | Path):
    """Test the integration of extracting mesh boundary vertices from a file."""
    local_mesh = mops.read_mesh(gltf_simple_file)
    edges = local_mesh.edges
    boundary_vertices = mops.get_boundary_vertices(edges)
    assert np.array_equiv(
        np.sort(boundary_vertices), [0, 2, 4, 6, 7, 9, 10]
    ), "The vertices returned by get_border_vertices do not match the expected vertices."


def test_get_basecolor_texture(gltf_file_with_basecolor_texture: str | Path, mock_image: npt.NDArray[Any]):
    """Test the integration of extracting an image from GLB."""
    local_mesh = mops.read_mesh(gltf_file_with_basecolor_texture)
    extracted_image = local_mesh.material.baseColorTexture
    assert np.array_equal(extracted_image, mock_image), "The extracted image does not match the expected image."


def test_access_material_should_fail(gltf_simple_file: str | Path):
    """Test the simple gltf does not have a material."""
    mesh = mops.read_mesh(gltf_simple_file)
    assert mesh.material is None, "Mesh should not have a material."

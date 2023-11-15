from pathlib import Path

import numpy as np

import readyplayerme.meshops.mesh as mops


def test_boundary_vertices_from_file(gltf_simple_file: str | Path):
    """Test the integration of extracting mesh boundary vertices from a file."""
    local_mesh = mops.read_mesh(gltf_simple_file)
    edges = local_mesh.edges
    boundary_vertices = mops.get_boundary_vertices(edges)
    assert np.array_equiv(
        np.sort(boundary_vertices), [0, 2, 4, 6, 7, 9, 10]
    ), "The vertices returned by get_border_vertices do not match the expected vertices."

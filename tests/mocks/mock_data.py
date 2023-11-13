"""Mock data for tests."""
from unittest.mock import create_autospec

import numpy as np
import trimesh

"""Edges values are extracted from uv-seams.glb"""
EDGES = [
    [9, 2],
    [2, 3],
    [3, 9],
    [3, 7],
    [7, 9],
    [9, 3],
    [7, 8],
    [8, 10],
    [10, 7],
    [8, 5],
    [5, 1],
    [1, 8],
    [3, 5],
    [5, 7],
    [7, 3],
    [1, 2],
    [2, 0],
    [0, 1],
    [3, 1],
    [1, 5],
    [5, 3],
    [9, 6],
    [6, 2],
    [2, 9],
    [1, 3],
    [3, 2],
    [2, 1],
    [7, 5],
    [5, 8],
    [8, 7],
    [8, 0],
    [0, 4],
    [4, 8],
    [10, 8],
    [8, 4],
    [4, 10],
    [8, 1],
    [1, 0],
    [0, 8],
]

mock_mesh_trimesh = create_autospec(trimesh.Trimesh, instance=True)

# Create a mock mesh object
mock_mesh_trimesh.edges = np.array(EDGES)

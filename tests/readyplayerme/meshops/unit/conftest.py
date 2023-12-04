"""Pytest fixtures for meshops unit tests."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture
def mock_mesh():
    """Return a mocked instance of a mesh."""

    @dataclass
    class MockMesh:
        vertices: npt.NDArray[np.float32]
        edges: npt.NDArray[np.int32]
        faces: npt.NDArray[np.int32]

    vertices = np.array(
        [
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
        ]
    )
    edges = np.array(
        [
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
    )
    faces = np.array(
        [
            [9, 2, 3],
            [3, 7, 9],
            [7, 8, 10],
            [8, 5, 1],
            [3, 5, 7],
            [1, 2, 0],
            [3, 1, 5],
            [9, 6, 2],
            [1, 3, 2],
            [7, 5, 8],
            [8, 0, 4],
            [10, 8, 4],
            [8, 1, 0],
        ]
    )
    return MockMesh(vertices=vertices, edges=edges, faces=faces)


@pytest.fixture
def mocked_image_diagonal_line_rgb():
    """Return an image with 1 diagonal line of lerped rgb colors."""
    return np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [170, 85, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [85, 170, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def mocked_image_diagonal_line_grayscale():
    """Return an image with 1 diagonal line of lerped grayscale colors."""
    return np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 255, 0, 0, 0, 0],
            [0, 0, 191, 0, 0, 0],
            [0, 0, 0, 127, 0, 0],
            [0, 0, 0, 0, 63, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

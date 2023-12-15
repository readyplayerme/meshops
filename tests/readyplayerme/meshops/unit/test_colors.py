import numpy as np
import pytest

from readyplayerme.meshops import draw
from readyplayerme.meshops.types import ColorMode


@pytest.mark.parametrize(
    "image, expected_mode",
    [
        # Grayscale image (2D array)
        (np.array([[0, 1], [1, 0]], dtype=np.float32), ColorMode.GRAYSCALE),
        # RGB image (3D array with shape (h, w, 3))
        (np.random.rand(10, 10, 3).astype(np.float32), ColorMode.RGB),
        # RGBA image (3D array with shape (h, w, 4))
        (np.random.rand(10, 10, 4).astype(np.float32), ColorMode.RGBA),
    ],
)
def test_get_image_color_mode(image, expected_mode):
    """Test the get_image_color_mode function with valid inputs."""
    assert draw.get_image_color_mode(image) == expected_mode


@pytest.mark.parametrize(
    "image",
    [
        # Invalid image: 2D array with incorrect channel count
        np.random.rand(10, 10, 5).astype(np.float32),
        # Invalid image: 1D array
        np.array([1, 2, 3], dtype=np.float32),
        # Invalid image: 4D array
        np.random.rand(10, 10, 10, 3).astype(np.float32),
    ],
)
def test_get_image_color_mode_should_fail(image):
    """Test the get_image_color_mode function with invalid inputs."""
    with pytest.raises(ValueError):
        draw.get_image_color_mode(image)


@pytest.mark.parametrize(
    "color_array, expected_mode",
    [
        # Grayscale color array (1D array)
        (np.array([128, 255, 100], dtype=np.uint8), ColorMode.GRAYSCALE),
        # Grayscale color array (2D array with single channel)
        (np.array([[128], [255], [100]], dtype=np.uint8), ColorMode.GRAYSCALE),
        # RGB color array (2D array with shape (n, 3))
        (np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8), ColorMode.RGB),
        # RGBA color array (2D array with shape (n, 4))
        (np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8), ColorMode.RGBA),
    ],
)
def test_get_color_array_color_mode(color_array, expected_mode):
    """Test the get_color_array_color_mode function with valid inputs."""
    assert draw.get_color_array_color_mode(color_array) == expected_mode


@pytest.mark.parametrize(
    "color_array",
    [
        # Invalid color array: 2D array with incorrect channel count
        np.array([[1, 2, 3, 4, 5]], dtype=np.uint8),
        # Invalid color array: 3D array
        np.random.rand(10, 10, 3).astype(np.uint8),
        # Invalid color array: 0-dimensional array
        np.array(128, dtype=np.uint8),
    ],
)
def test_get_color_array_color_mode_should_fail(color_array):
    """Test the get_color_array_color_mode function with invalid inputs."""
    with pytest.raises(ValueError):
        draw.get_color_array_color_mode(color_array)


@pytest.mark.parametrize(
    "colors, index_groups, expected",
    [
        # Case with simple groups and distinct colors
        (
            np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]),
            [[0, 1], [2, 3]],
            np.array([[127, 127, 0], [127, 127, 0], [127, 127, 127], [127, 127, 127]]),
        ),
        # Case with a single group
        (
            np.array([[100, 100, 100], [200, 200, 200]]),
            [[0, 1]],
            np.array([[150, 150, 150], [150, 150, 150]]),
        ),
        # Case with groups of 1 element
        (
            np.array([[100, 0, 0], [0, 100, 0], [0, 0, 100]]),
            [[0], [1], [2]],
            np.array([[100, 0, 0], [0, 100, 0], [0, 0, 100]]),
        ),
        # Case with empty colors array
        (np.array([], dtype=np.uint8), [[0, 1]], np.array([])),
        # Case with empty groups
        (np.array([[255, 0, 0], [0, 255, 0]]), [], np.array([[255, 0, 0], [0, 255, 0]])),
        # Case with empty colors and groups
        (np.array([], dtype=np.uint8), [], np.array([], dtype=np.uint8)),
    ],
)
def test_blend_colors(colors, index_groups, expected):
    """Test the blend_colors function."""
    blended_colors = draw.blend_colors(colors, index_groups)
    np.testing.assert_array_equal(blended_colors, expected)


@pytest.mark.parametrize(
    "colors, index_groups",
    [
        # Case with out-of-bounds indices
        (np.array([[255, 0, 0], [0, 255, 0]]), [[0, 2]]),
        # Case with negative index
        (np.array([[255, 0, 0], [0, 255, 0]]), [[-3, 1]]),
    ],
)
def test_blend_colors_should_fail(colors, index_groups):
    """Test error handling in blend_colors function."""
    with pytest.raises(IndexError):
        draw.blend_colors(colors, index_groups)


@pytest.mark.parametrize(
    "start_color, end_color, num_steps, expected_output",
    [
        # Basic interpolation from red to blue
        (
            np.array([255, 0, 0]),
            np.array([0, 0, 255]),
            3,
            np.array([[255.0, 0.0, 0.0], [127.5, 0.0, 127.5], [0.0, 0.0, 255.0]]),
        ),
        # Interpolation from RGBA red to RGBA blue (with alpha channel)
        (
            np.array([255, 0, 0, 0.5]),  # Start color: Red with half opacity
            np.array([0, 0, 255, 1.0]),  # End color: Blue with full opacity
            3,
            np.array([[255.0, 0.0, 0.0, 0.5], [127.5, 0.0, 127.5, 0.75], [0.0, 0.0, 255.0, 1.0]]),
        ),
        # Interpolation in grayscale
        (
            np.array([0]),  # Start color: Black in grayscale
            np.array([255]),  # End color: White in grayscale
            3,
            np.array([0, 127.5, 255]),  # Intermediate grayscale values
        ),
        # Interpolation with more steps
        (
            np.array([255, 0, 0]),
            np.array([0, 255, 0]),
            5,
            np.array(
                [[255.0, 0.0, 0.0], [191.25, 63.75, 0.0], [127.5, 127.5, 0.0], [63.75, 191.25, 0.0], [0.0, 255.0, 0.0]]
            ),
        ),
        # Interpolation with a single step (should return start_color)
        (np.array([255, 255, 0]), np.array([0, 255, 255]), 1, np.array([[255, 255, 0]])),
        # Interpolation with two steps (should return start and end colors only)
        (np.array([0, 255, 0]), np.array([0, 0, 255]), 2, np.array([[0, 255, 0], [0, 0, 255]])),
    ],
)
def test_interpolate_values(start_color, end_color, num_steps, expected_output):
    """Test the vectorized_interpolate function with various input scenarios."""
    actual_output = draw.interpolate_values(start_color, end_color, num_steps)
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=5)


@pytest.mark.parametrize(
    "start_color, end_color, num_steps, expected_exception",
    [
        # Mismatched shapes
        (np.array([255, 0, 0]), np.array([0, 0]), 3, ValueError),
        # Invalid number of steps (less than 1)
        (np.array([255, 0, 0]), np.array([0, 0, 255]), 0, ValueError),
        # Invalid type for start or end color
        ("255, 0, 0", np.array([0, 0, 255]), 3, AttributeError),
    ],
)
def test_interpolate_values_should_fail(start_color, end_color, num_steps, expected_exception):
    """Test the vectorized_interpolate function with invalid input scenarios."""
    with pytest.raises(expected_exception):
        draw.interpolate_values(start_color, end_color, num_steps)

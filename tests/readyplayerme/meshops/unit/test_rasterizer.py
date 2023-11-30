"""Unit tests for the rasterize module."""

import typing

import numpy as np
import pytest

import readyplayerme.meshops.draw.rasterize as rast
from readyplayerme.meshops.types import Color, Edges, Image, PixelCoord


@pytest.mark.parametrize(
    "input_segment, expected_output",
    [
        # All NaNs
        (np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])),
        # No NaNs
        (np.array([1, 2, 3]), np.array([1, 2, 3])),
        # Single Element
        (np.array([1]), np.array([1])),
        # Single NaN
        (np.array([np.nan]), np.array([np.nan])),
    ],
)
def test_interpolate_segment(input_segment, expected_output):
    """Test the interpolate_segment function with various input scenarios."""
    output = rast.interpolate_segment(input_segment)
    np.testing.assert_array_equal(output, expected_output)


@pytest.mark.parametrize(
    "image_array, edges, image_coords, colors, interpolate_func, expected_output",
    [
        # Empty Edges with Mocked Data
        (
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.array([]),
            np.array([[0, 0], [1, 1]]),
            np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
            lambda color0, color1, steps: np.array([[100, 100, 100]] * steps, dtype=np.uint8),  # noqa: ARG005
            np.zeros((5, 5, 3), dtype=np.uint8),
        ),
        # Non-Existent Edge Points with Mocked Data
        (
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.array([[0, 2]]),  # Edge from point 0 to point 2
            np.array([[0, 0], [1, 1], [4, 4]]),  # Coordinates for the points
            np.array([[255, 0, 0], [0, 255, 0], [200, 50, 50]], dtype=np.uint8),  # Colors for the points
            lambda color0, color1, steps: np.array([[200, 50, 50]] * steps, dtype=np.uint8),  # noqa: ARG005
            np.array(
                [  # Expected output with a line drawn
                    [[200, 50, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [200, 50, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [200, 50, 50], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [200, 50, 50], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [200, 50, 50]],
                ],
                dtype=np.uint8,
            ),
        ),
        # Zero Length Lines with Mocked Data
        (
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.array([[0, 0]]),  # Start and end points are the same
            np.array([[0, 0], [1, 1]]),  # Coordinates for the points
            np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),  # Colors for the points
            lambda color0, color1, steps: np.array([[50, 200, 200]] * steps, dtype=np.uint8),  # noqa: ARG005
            np.array(
                [  # Expected output with a single pixel drawn
                    [[50, 200, 200], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
                dtype=np.uint8,
            ),
        ),
    ],
)
def test_draw_lines(image_array, edges, image_coords, colors, interpolate_func, expected_output):
    """Test draw_lines function with various edge cases."""
    output = rast.draw_lines(image_array, edges, image_coords, colors, interpolate_func)
    np.testing.assert_array_equal(output, expected_output)


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
def test_vectorized_interpolate(start_color, end_color, num_steps, expected_output):
    """Test the vectorized_interpolate function with various input scenarios."""
    actual_output = rast.vectorized_interpolate(start_color, end_color, num_steps)
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
def test_vectorized_interpolate_failures(start_color, end_color, num_steps, expected_exception):
    """Test the vectorized_interpolate function with invalid input scenarios."""
    with pytest.raises(expected_exception):
        rast.vectorized_interpolate(start_color, end_color, num_steps)


@pytest.mark.parametrize(
    "input_array,expected_output",
    [
        # Interpolate NaNs in-between valid values
        (np.array([[1, np.nan, 3, np.nan, 5]]), np.array([[1, 2, 3, 4, 5]])),
        # Horizontal interpolation with multiple columns
        (
            np.array([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 7, 8]]),
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 5],
                    [7, 7, 8],
                ]
            ),
        ),
        # Multiple NaNs in-between
        (np.array([[1, np.nan, np.nan, 4, 5]]), np.array([[1, 2, 3, 4, 5]])),
        # No NaNs in-between (no interpolation needed)
        (np.array([[1, 2, 3, 4, 5]]), np.array([[1, 2, 3, 4, 5]])),
        # NaNs only at the edges should remain as NaNs
        (np.array([[np.nan, 1, 2, 3, np.nan]]), np.array([[1, 1, 2, 3, 3]])),
        # Single NaN in-between
        (np.array([[1, 2, np.nan, 4, 5]]), np.array([[1, 2, 3, 4, 5]])),
        # All NaNs except edges
        (np.array([[1, np.nan, np.nan, np.nan, 5]]), np.array([[1, 2, 3, 4, 5]])),
        # Single-Row Array with NaN in-between
        (np.array([1, np.nan, 3]), np.array([[1, 2, 3]])),
        # Empty Arrays
        (np.array([[]]), np.array([[]])),
        # Single Element Arrays
        (np.array([[1]]), np.array([[1]])),
        # Arrays with No NaN Values
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]])),
        # All NaN Arrays
        (np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.array([[np.nan, np.nan], [np.nan, np.nan]])),
    ],
)
def test_vectorized_lerp_nans_vertically(input_array, expected_output):
    """Test vectorized_lerp_nans_vertically function with various input scenarios."""
    actual_output = rast.vectorized_lerp_nans_vertically(input_array)
    np.testing.assert_array_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    "input_array,expected_output",
    [
        # Basic vertical interpolation single columns
        (np.array([[1], [np.nan], [3]]), np.array([[1], [2], [3]])),
        # Basic vertical interpolation multiple columns
        (
            np.array([[1, np.nan, 2], [np.nan, 3, np.nan], [3, np.nan, 4]]),
            np.array([[1, 3, 2], [2, 3, 3], [3, 3, 4]]),
        ),
        # Edge cases
        (np.array([[np.nan], [2], [3]]), np.array([[2], [2], [3]])),
        # Multiple columns with nan esges
        (np.array([[1], [2], [np.nan]]), np.array([[1], [2], [2]])),
        (
            np.array([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 7, 5]]),
            np.array(
                [
                    [1, 5, 3],
                    [4, 5, 4],
                    [4, 7, 5],
                ]
            ),
        ),
        # Multiple consecutive NaNs
        (np.array([[1], [np.nan], [np.nan], [4]]), np.array([[1], [2], [3], [4]])),
        # No NaNs
        (np.array([[1], [2], [3]]), np.array([[1], [2], [3]])),
        # All NaNs
        (np.array([[np.nan], [np.nan], [np.nan]]), np.array([[np.nan], [np.nan], [np.nan]])),
        # Single-column Array
        (np.array([1, np.nan, 3]), np.array([[1], [2], [3]])),
        # Empty Arrays
        (np.array([]), np.array([]).reshape(0, 1)),
        # Single Element Arrays
        (np.array([[1]]), np.array([[1]])),
        # Arrays with No NaN Values
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]])),
        # All NaN Arrays
        (np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.array([[np.nan, np.nan], [np.nan, np.nan]])),
    ],
)
def test_vectorized_lerp_nans_horizontally(input_array, expected_output):
    """Test vectorized_lerp_nans_horizontally function with various input scenarios."""
    actual_output = rast.vectorized_lerp_nans_horizontally(input_array)
    np.testing.assert_array_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    "width, height, edges, image_coords, colors, interpolate_func, fill_func, expected_output",
    [
        # Basic Functionality
        (
            6,
            6,
            np.array([[0, 1]]),
            np.array([[1, 1], [4, 4]]),
            np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
            lambda color0, color1, steps: np.linspace(color0, color1, steps).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            "mock_image",
        ),
        # No Edges
        (
            50,
            50,
            np.array([]),
            np.array([]),
            np.array([]),
            lambda color0, color1, steps: np.linspace(color0, color1, steps).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            np.zeros((50, 50, 3), dtype=np.uint8),  # Expected blank image
        ),
        # Single Pixel Image
        (
            1,
            1,
            np.array([[0, 0]]),
            np.array([[0, 0]]),
            np.array([[255, 0, 0]]),
            lambda color0, color1, steps: np.linspace(color0, color1, steps).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            np.array([[[255, 0, 0]]], dtype=np.float32),  # Expected output
        ),
        # More test cases...
    ],
)
def test_rasterize_valid(
    width: int,
    height: int,
    edges: Edges,
    image_coords: PixelCoord,
    colors: Color,
    interpolate_func: typing.Callable[[Color, Color, int], Color],
    fill_func: typing.Callable[[Image], Image],
    expected_output: Image,
    request: pytest.FixtureRequest,
):
    """Test rasterize function with valid inputs."""
    if isinstance(expected_output, str) and expected_output == "mock_image":
        expected_output = request.getfixturevalue("mock_image").mock_rasterized_image
    output = rast.rasterize(
        width,
        height,
        edges,
        image_coords,
        colors,
        interpolate_func,
        fill_func,
    )

    np.testing.assert_array_equal(output, expected_output, err_msg="Rasterized image did not match expected output.")


@pytest.mark.parametrize(
    "width, height, edges, image_coords, all_vertex_colors, interpolate_func, fill_func, expected_exception",
    [
        # Negative Dimensions
        (
            -100,
            100,
            np.array([[0, 1]]),
            np.array([[0, 0], [10, 10]]),
            np.array([[255, 0, 0], [0, 255, 0]]),
            lambda color0, color1, steps: np.linspace(color0, color1, steps).astype(np.uint8),  # Mock interpolate func
            lambda img: np.nan_to_num(img).astype(np.uint8),  # Mock fill func
            ValueError,
        ),
        # Out of Bounds Edges
        (
            100,
            100,
            np.array([[0, 10]]),
            np.array([[10, 10]]),
            np.array([[255, 0, 0]]),
            lambda color0, color1, steps: np.linspace(color0, color1, steps).astype(np.uint8),  # Mock interpolate func
            lambda img: np.nan_to_num(img).astype(np.uint8),  # Mock fill func
            IndexError,
        ),
        # Zero Dimensions
        (
            0,
            0,
            np.array([[0, 1]]),
            np.array([[0, 0], [10, 10]]),
            np.array([[255, 0, 0], [0, 255, 0]]),
            lambda color0, color1, steps: np.linspace(color0, color1, steps).astype(np.uint8),  # Mock interpolate func
            lambda img: np.nan_to_num(img).astype(np.uint8),  # Mock fill func
            IndexError,
        ),
        # More test cases...
    ],
)
def test_rasterize_should_fail(
    width, height, edges, image_coords, all_vertex_colors, interpolate_func, fill_func, expected_exception
):
    """Test rasterize function with invalid inputs."""
    with pytest.raises(expected_exception):
        rast.rasterize(
            width,
            height,
            edges,
            image_coords,
            all_vertex_colors,
            interpolate_func,
            fill_func,
        )

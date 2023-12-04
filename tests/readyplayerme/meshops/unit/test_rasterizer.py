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
        # Interpolation with NaNs in the middle
        (np.array([1, np.nan, 3]), np.array([1, 2, 3])),
        # Interpolation with multiple NaNs
        (np.array([1, np.nan, np.nan, 4]), np.array([1, 2, 3, 4])),
        # NaN at the beginning
        (np.array([np.nan, 2, 3]), np.array([2, 2, 3])),
        # NaN at the end
        (np.array([1, 2, np.nan]), np.array([1, 2, 2])),
        # NaNs at both ends
        (np.array([np.nan, 2, np.nan]), np.array([2, 2, 2])),
    ],
)
def test_interpolate_segment(input_segment, expected_output):
    """Test the interpolate_segment function with various input scenarios."""
    output = rast.interpolate_segment(input_segment)
    np.testing.assert_array_equal(output, expected_output)


@pytest.mark.parametrize(
    "image, edges, image_coords, colors, interpolate_func, expected_output",
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
        # Test with RGBA image
        (
            np.zeros((5, 5, 4), dtype=np.uint8),  # RGBA image array
            np.array([[0, 1]]),  # Edge from point 0 to point 1
            np.array([[0, 0], [4, 4]]),  # Coordinates for the points
            np.array([[255, 0, 0, 128], [0, 255, 0, 255]], dtype=np.uint8),  # Colors for the points (RGBA)
            lambda color0, color1, steps: np.array([[127, 127, 0, 191]] * steps, dtype=np.uint8),  # noqa: ARG005
            np.array(
                [
                    [[127, 127, 0, 191], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [127, 127, 0, 191], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [127, 127, 0, 191], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [127, 127, 0, 191], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [127, 127, 0, 191]],
                ],
                dtype=np.uint8,
            ),
        ),
        # Test with grayscale image
        (
            np.zeros((5, 5), dtype=np.uint8),  # Grayscale image array
            np.array([[0, 1]]),  # Edge from point 0 to point 1
            np.array([[0, 0], [4, 4]]),  # Coordinates for the points
            np.array([[0], [255]], dtype=np.uint8),  # Colors for the points (grayscale)
            lambda color0, color1, steps: np.array([128] * steps, dtype=np.uint8),  # noqa: ARG005
            np.array(
                [
                    [128, 0, 0, 0, 0],
                    [0, 128, 0, 0, 0],
                    [0, 0, 128, 0, 0],
                    [0, 0, 0, 128, 0],
                    [0, 0, 0, 0, 128],
                ],
                dtype=np.uint8,
            ),
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
def test_draw_lines(image, edges, image_coords, colors, interpolate_func, expected_output):
    """Test draw_lines function with various edge cases."""
    output = rast.draw_lines(image, edges, image_coords, colors, interpolate_func)
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
            np.array([[0], [127.5], [255]]),  # Intermediate grayscale values
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
    actual_output = rast.interpolate_values(start_color, end_color, num_steps)
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
        rast.interpolate_values(start_color, end_color, num_steps)


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
def test_lerp_nans_horizontally(input_array, expected_output):
    """Test vectorized_lerp_nans_vertically function with various input scenarios."""
    actual_output = rast.lerp_nans_horizontally(input_array)
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
def test_lerp_nans_vertically(input_array, expected_output):
    """Test vectorized_lerp_nans_horizontally function with various input scenarios."""
    actual_output = rast.lerp_nans_vertically(input_array)
    np.testing.assert_array_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    "width, height",
    [
        (100, 100),  # Typical usage
    ],
)
def test_create_nan_image(width, height):
    """Test the create_nan_image function with valid inputs."""
    result = rast.create_nan_image(width, height)
    assert result.shape == (height, width, 3)
    assert np.isnan(result).all()


@pytest.mark.parametrize(
    "width, height, error",
    [
        (0, 100, ValueError),  # Zero width
        (100, 0, ValueError),  # Zero height
        (-100, 100, ValueError),  # Negative width
        (100, -100, ValueError),  # Negative height
        (100.5, 100, TypeError),  # Float width
        (100, 100.5, TypeError),  # Float height
    ],
)
def test_create_nan_image_should_fail(width, height, error):
    """Test the create_nan_image function with invalid inputs."""
    with pytest.raises(error):
        rast.create_nan_image(width, height)


@pytest.mark.parametrize(
    "input_array, expected_output",
    [
        # No NaNs or Infinities
        (np.array([[100, 150], [200, 250]], dtype=np.float32), np.array([[100, 150], [200, 250]], dtype=np.float32)),
        # Contains NaNs
        (np.array([[np.nan, 150], [200, np.nan]], dtype=np.float32), np.array([[0, 150], [200, 0]], dtype=np.float32)),
        # Contains Positive and Negative Infinities
        (
            np.array([[np.inf, -np.inf], [200, 300]], dtype=np.float32),
            np.array([[255, 0], [200, 255]], dtype=np.float32),
        ),
        # Mix of NaNs and Infinities
        (
            np.array([[np.nan, -np.inf], [np.inf, np.nan]], dtype=np.float32),
            np.array([[0, 0], [255, 0]], dtype=np.float32),
        ),
        # Values Exceeding the Range [0, 255]
        (np.array([[300, -100], [500, 600]], dtype=np.float32), np.array([[255, 0], [255, 255]], dtype=np.float32)),
    ],
)
def test_clean_image(input_array, expected_output):
    """Test the clean_image function with various input scenarios."""
    output = rast.clean_image(input_array, inplace=False)
    np.testing.assert_array_equal(output, expected_output)


@pytest.mark.parametrize(
    "image, edges, image_coords, colors, interpolate_func, fill_func, expected_output",
    [
        # Basic Functionality
        (
            np.full((6, 6, 3), np.nan, dtype=np.float32),
            np.array([[0, 1]]),
            np.array([[1, 1], [4, 4]]),
            np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
            lambda c0, c1, steps: np.linspace(c0, c1, steps).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            "mock_image",
        ),
        # No Edges
        (
            np.full((50, 50, 3), np.nan, dtype=np.float32),
            np.array([]),
            np.array([]),
            np.array([]),
            lambda c0, c1, steps: np.linspace(c0, c1, steps).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            np.zeros((50, 50, 3), dtype=np.uint8),
        ),
        # Single Pixel Image
        (
            np.full((1, 1, 3), np.nan, dtype=np.float32),
            np.array([[0, 0]]),
            np.array([[0, 0]]),
            np.array([[255, 0, 0]]),
            lambda c0, c1, steps: np.linspace(c0, c1, steps).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            np.array([[[255, 0, 0]]], dtype=np.float32),
        ),
    ],
)
def test_rasterize(
    image: Image,
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
        expected_output = request.getfixturevalue("mocked_image_diagonal_line_rgb")
    output = rast.rasterize(
        image,
        edges,
        image_coords,
        colors,
        interpolate_func,
        fill_func,
    )

    np.testing.assert_array_equal(output, expected_output, err_msg="Rasterized image did not match expected output.")


@pytest.mark.parametrize(
    "image, edges, image_coords, colors, interpolate_func, fill_func, expected_exception",
    [
        # Out of Bounds Edges
        (
            np.full((100, 100, 3), np.nan, dtype=np.float32),
            np.array([[0, 10]]),
            np.array([[10, 10]]),
            np.array([[255, 0, 0]]),
            lambda color0, color1, steps: np.linspace(color0, color1, steps).astype(np.uint8),  # Mock interpolate func
            lambda img: np.nan_to_num(img).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            IndexError,
        ),
        # Zero Dimensions
        (
            np.full((0, 0, 3), np.nan, dtype=np.float32),
            np.array([[0, 1]]),
            np.array([[0, 0], [10, 10]]),
            np.array([[255, 0, 0], [0, 255, 0]]),
            lambda color0, color1, steps: np.linspace(color0, color1, steps).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            IndexError,
        ),
        # Mismatched Array Sizes
        (
            np.full((10, 10, 3), np.nan, dtype=np.float32),
            np.array([[0, 1], [2, 3]]),  # Two edges defined
            np.array([[0, 0], [1, 1]]),  # Only two coordinates provided
            np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),  # Three colors provided
            lambda c0, c1, steps: np.linspace(c0, c1, steps).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            IndexError,
        ),
    ],
)
def test_rasterize_should_fail(
    image: Image,
    edges: Edges,
    image_coords: PixelCoord,
    colors: Color,
    interpolate_func: typing.Callable[[Color, Color, int], Color],
    fill_func: typing.Callable[[Image], Image],
    expected_exception: type[Exception],
):
    """Test rasterize function with invalid inputs."""
    with pytest.raises(expected_exception):
        rast.rasterize(image, edges, image_coords, colors, interpolate_func, fill_func)

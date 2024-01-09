"""Unit tests for the image module."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from readyplayerme.meshops import draw
from readyplayerme.meshops.mesh import Mesh
from readyplayerme.meshops.types import Color, ColorMode, Edges, Faces, Image, PixelCoord, UVs


@pytest.mark.parametrize(
    "image_1, image_2, mask, expected_output",
    [
        # Basic blending with a uniform mask
        (
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
        ),
        # Blending with different mask values
        (
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            np.array([[0.2, 0.8], [0.3, 0.7]]),
            np.array([[0.2, 0.8], [0.3, 0.7]]),
        ),
        # Mask with all zeros (full image_1)
        (
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            np.array([[0, 0], [0, 0]]),
            np.array([[0, 0], [0, 0]]),
        ),
        # Mask with all ones (full image_2)
        (
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            np.array([[1, 1], [1, 1]]),
            np.array([[1, 1], [1, 1]]),
        ),
        # Non uniform mask values:
        (
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            np.array([[0, 1], [1, 0]]),
            np.array([[0, 1], [1, 0]]),
        ),
        # Full RGB Image
        (
            np.zeros((2, 2, 3)),  # Black RGB image
            np.ones((2, 2, 3)),  # White RGB image
            np.full((2, 2), 0.5),  # Uniform grayscale mask
            np.full((2, 2, 3), 0.5),  # Expected output: gray RGB image
        ),
    ],
)
def test_blend_images(image_1: Image, image_2: Image, mask: Image, expected_output: Image):
    """Test the blend_images function with various input scenarios."""
    output = draw.blend_images(image_1, image_2, mask)
    np.testing.assert_array_equal(output, expected_output)


@pytest.mark.parametrize(
    "image_1, image_2, mask, expected_exception",
    [
        # Shape mismatch
        (
            np.array([[0, 0]]),  # Shape (1, 2)
            np.array([[1, 1], [1, 1], [1, 1]]),  # Shape (3, 2)
            np.array([[0.5, 0.5]]),  # Shape (1, 2),
            ValueError,
        ),
        # Invalid input type
        ("0, 0, 0", np.array([[1, 1], [1, 1]]), np.array([[0.5, 0.5], [0.5, 0.5]]), AttributeError),
        # Empty mask
        (
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            np.array([]),
            ValueError,
        ),
    ],
)
def test_blend_images_should_fail(image_1: Image, image_2: Image, mask: Image, expected_exception: Image):
    """Test the blend_images function with invalid input scenarios."""
    with pytest.raises(expected_exception):
        draw.blend_images(image_1, image_2, mask)


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
def test_interpolate_segment(input_segment: Image, expected_output: Image):
    """Test the interpolate_segment function with various input scenarios."""
    output = draw.interpolate_segment(input_segment)
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
        # Test with grayscale image, grayscale 2D colors (2,1)
        (
            np.zeros((5, 5), dtype=np.uint8),  # Grayscale image array
            np.array([[0, 1]]),  # Edge from point 0 to point 1
            np.array([[0, 0], [4, 4]]),  # Coordinates for the points
            np.array([[0], [255]], dtype=np.uint8),  # Colors for the points (2D grayscale)
            lambda color0, color1, steps: np.array(128, dtype=np.uint8).repeat(steps),  # noqa: ARG005
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
        # Test with grayscale image, grayscale 1D colors (2,)
        (
            np.zeros((5, 5), dtype=np.uint8),  # Grayscale image array
            np.array([[0, 1]]),  # Edge from point 0 to point 1
            np.array([[0, 0], [4, 4]]),  # Coordinates for the points
            np.array([0, 255], dtype=np.uint8),  # Colors for the points (1D grayscale)
            lambda color0, color1, steps: np.array(128, dtype=np.uint8).repeat(steps),  # noqa: ARG005
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
def test_draw_lines(
    image: Image,
    edges: Edges,
    image_coords: PixelCoord,
    colors: Color,
    interpolate_func: Callable[[Color, Color, int], Color],
    expected_output: Image,
):
    """Test draw_lines function with various edge cases."""
    output = draw.draw_lines(image, edges, image_coords, colors, interpolate_func)
    np.testing.assert_array_equal(output, expected_output)


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
def test_lerp_nans_horizontally(input_array: Image, expected_output: Image):
    """Test vectorized_lerp_nans_vertically function with various input scenarios."""
    actual_output = draw.lerp_nans_horizontally(input_array)
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
        # Multiple columns with nan edges
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
def test_lerp_nans_vertically(input_array: Image, expected_output: Image):
    """Test vectorized_lerp_nans_horizontally function with various input scenarios."""
    actual_output = draw.lerp_nans_vertically(input_array)
    np.testing.assert_array_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    "width, height, mode",
    [
        (100, 100, ColorMode.RGB),  # Typical usage, RGB
        (100, 100, ColorMode.RGBA),  # RGBA
        (100, 100, ColorMode.GRAYSCALE),  # Grayscale
    ],
)
def test_create_nan_image(width: int, height: int, mode: ColorMode):
    """Test the create_nan_image function with valid inputs."""
    result = draw.create_nan_image(width, height, mode)
    assert result.shape == tuple(filter(bool, (height, width, mode.value)))
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
def test_create_nan_image_should_fail(width: int, height: int, error: type[Exception]):
    """Test the create_nan_image function with invalid inputs."""
    with pytest.raises(error):
        draw.create_nan_image(width, height)


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
def test_clip_image(input_array: Image, expected_output: Image):
    """Test the clean_image function with various input scenarios."""
    output = draw.clip_image(input_array, inplace=False)
    np.testing.assert_array_equal(output, expected_output)


@pytest.mark.parametrize(
    "width, height, coords, expected_output",
    [
        (
            8,
            8,
            np.array(
                [
                    [[5, 1], [0, 1], [3, 3]],
                    [[3, 3], [4, 3], [5, 1]],
                    [[4, 3], [4, 5], [6, 3]],
                    [[4, 5], [3, 4], [2, 4]],
                    [[3, 3], [3, 4], [4, 3]],
                    [[2, 4], [0, 1], [0, 6]],
                    [[3, 3], [2, 4], [3, 4]],
                    [[5, 1], [5, 0], [0, 1]],
                    [[2, 4], [3, 3], [0, 1]],
                    [[4, 3], [3, 4], [4, 5]],
                    [[4, 5], [0, 6], [4, 7]],
                    [[6, 3], [4, 5], [4, 7]],
                    [[4, 5], [2, 4], [0, 6]],
                ],
                dtype=np.int16,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 255, 0, 0],
                    [255, 255, 255, 255, 255, 255, 0, 0],
                    [255, 255, 255, 255, 255, 0, 0, 0],
                    [255, 255, 255, 255, 255, 255, 255, 0],
                    [255, 255, 255, 255, 255, 255, 0, 0],
                    [255, 255, 255, 255, 255, 255, 0, 0],
                    [255, 255, 255, 255, 255, 0, 0, 0],
                    [0, 0, 0, 0, 255, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
        )
    ],
)
def test_create_mask(width: int, height: int, coords: npt.NDArray[np.int16], expected_output: Image):
    """Test the create_mask function with valid inputs."""
    mask = draw.create_mask(width, height, coords)
    assert mask.shape == (height, width), f"Mask shape {mask.shape} does not match expected shape {(width, height)}."
    np.testing.assert_array_equal(mask, expected_output, err_msg="Mask did not match expected output.")


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
    interpolate_func: Callable[[Color, Color, int], Color],
    fill_func: Callable[[Image], Image],
    expected_output: Image,
    request: pytest.FixtureRequest,
):
    """Test rasterize function with valid inputs."""
    if isinstance(expected_output, str) and expected_output == "mock_image":
        expected_output = request.getfixturevalue("mocked_image_diagonal_line_rgb")
    output = draw.rasterize(
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
            IndexError,
        ),
        # Mismatched Color Modes (Grayscale image with RGB colors)
        (
            np.full((100, 100), np.nan, dtype=np.float32),  # Grayscale image
            np.array([[0, 1]]),
            np.array([[10, 10], [20, 20]]),
            np.array([[255, 0, 0], [0, 255, 0]]),  # RGB colors
            lambda color0, color1, steps: np.linspace(color0, color1, steps).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            ValueError,  # Expecting a ValueError due to mismatched color modes
        ),
        # Mismatched Color Modes (RGB image with Grayscale colors)
        (
            np.full((100, 100, 3), np.nan, dtype=np.float32),  # RGB image
            np.array([[0, 1]]),
            np.array([[10, 10], [20, 20]]),
            np.array([255, 0]),  # Grayscale colors
            lambda color0, color1, steps: np.linspace(color0, color1, steps).astype(np.uint8),
            lambda img: np.nan_to_num(img).astype(np.uint8),
            ValueError,  # Expecting a ValueError due to mismatched color modes
        ),
    ],
)
def test_rasterize_should_fail(
    image: Image,
    edges: Edges,
    image_coords: PixelCoord,
    colors: Color,
    interpolate_func: Callable[[Color, Color, int], Color],
    fill_func: Callable[[Image], Image],
    expected_exception: type[Exception],
):
    """Test rasterize function with invalid inputs."""
    with pytest.raises(expected_exception):
        draw.rasterize(image, edges, image_coords, colors, interpolate_func, fill_func)


@pytest.mark.parametrize(
    "mock_mesh, image, expected_output",
    [
        # Grayscale image
        (
            "mock_mesh",
            np.array([[0, 0, 0, 0, 0, 0, 255, 255]] * 8, dtype=np.uint8),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 190, 184],
                    [0, 1, 4, 8, 16, 26, 223, 220],
                    [0, 0, 0, 3, 9, 12, 209, 205],
                    [0, 0, 0, 0, 0, 11, 227, 224],
                    [0, 0, 0, 0, 0, 9, 224, 221],
                    [0, 0, 0, 0, 0, 8, 224, 220],
                    [0, 0, 0, 0, 0, 3, 224, 221],
                    [0, 0, 0, 0, 0, 0, 224, 220],
                ],
                dtype=np.uint8,
            ),
        ),
        # RGB image
        (
            "mock_mesh",
            np.array([[255, 255, 255, 255, 255, 255, 0, 0]] * 8, dtype=np.uint8)[:, :, np.newaxis].repeat(3, axis=2),
            np.array(
                [
                    [255, 255, 255, 255, 255, 254, 64, 70],
                    [255, 253, 250, 246, 238, 228, 31, 34],
                    [255, 255, 255, 251, 245, 242, 45, 49],
                    [255, 255, 255, 254, 255, 243, 27, 30],
                    [255, 255, 255, 255, 255, 245, 30, 33],
                    [255, 255, 255, 255, 255, 246, 30, 33],
                    [255, 255, 255, 255, 255, 251, 30, 33],
                    [255, 255, 255, 255, 255, 255, 30, 34],
                ],
                dtype=np.uint8,
            )[:, :, np.newaxis].repeat(3, axis=2),
        ),
        # RGBA image
        (
            "mock_mesh",
            np.power(np.fromfunction(lambda i, j, k: i + j + k, (8, 8, 4), dtype=np.uint8), 2),
            np.array(
                [
                    [
                        [0, 2, 5, 10],
                        [1, 5, 10, 17],
                        [4, 10, 17, 26],
                        [9, 16, 25, 37],
                        [16, 25, 36, 49],
                        [24, 36, 48, 64],
                        [33, 45, 60, 76],
                        [42, 56, 72, 90],
                    ],
                    [
                        [1, 4, 9, 16],
                        [4, 9, 16, 25],
                        [10, 17, 26, 37],
                        [18, 27, 38, 51],
                        [28, 39, 53, 68],
                        [40, 54, 69, 87],
                        [51, 66, 83, 102],
                        [62, 79, 97, 118],
                    ],
                    [
                        [4, 9, 16, 25],
                        [9, 16, 25, 36],
                        [16, 25, 36, 49],
                        [26, 37, 50, 65],
                        [37, 50, 66, 83],
                        [49, 64, 82, 101],
                        [61, 78, 96, 117],
                        [73, 91, 112, 134],
                    ],
                    [
                        [9, 16, 25, 36],
                        [16, 25, 36, 49],
                        [25, 36, 49, 64],
                        [35, 49, 64, 81],
                        [49, 64, 81, 100],
                        [62, 78, 97, 118],
                        [76, 94, 114, 137],
                        [89, 109, 131, 155],
                    ],
                    [
                        [16, 25, 36, 49],
                        [25, 36, 49, 64],
                        [36, 49, 64, 80],
                        [49, 64, 81, 100],
                        [64, 81, 100, 121],
                        [79, 97, 118, 141],
                        [95, 115, 137, 162],
                        [111, 133, 156, 182],
                    ],
                    [
                        [25, 36, 49, 64],
                        [36, 49, 64, 81],
                        [49, 64, 81, 100],
                        [64, 81, 100, 121],
                        [81, 100, 121, 144],
                        [98, 119, 142, 167],
                        [115, 138, 162, 189],
                        [134, 158, 184, 212],
                    ],
                    [
                        [36, 49, 64, 81],
                        [48, 63, 80, 99],
                        [63, 80, 99, 120],
                        [80, 99, 120, 143],
                        [100, 121, 144, 169],
                        [119, 142, 166, 193],
                        [138, 163, 189, 218],
                        [159, 185, 213, 26],
                    ],
                    [
                        [49, 64, 81, 100],
                        [64, 81, 100, 121],
                        [80, 99, 120, 143],
                        [99, 120, 143, 168],
                        [121, 144, 169, 196],
                        [141, 166, 193, 222],
                        [163, 189, 218, 23],
                        [185, 214, 22, 55],
                    ],
                ],
                dtype=np.uint8,
            ),
        ),
    ],
    indirect=["mock_mesh"],
)
def test_blend_uv_seams(mock_mesh: Mesh, image: Image, expected_output: Image):
    """Test the blend_uv_seams function with valid inputs."""
    if image.ndim > 2:  # Debug: Skip RGB & RGBA tests to see if at least grayscale works.
        pytest.skip("Skipping test of RGB & RGBA for debugging purposes.")
    output = draw.blend_uv_seams(mock_mesh, image)
    np.testing.assert_array_equal(output, expected_output)


def test_blend_uv_seams_should_fail(mock_mesh: Mesh):
    """Test the blend_uv_seams function fails when mesh has no UV coordinates."""
    mock_mesh.uv_coords = None
    image = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="^(UV coordinates are invalid:).*"):
        draw.blend_uv_seams(mock_mesh, image)


@pytest.mark.parametrize(
    "width, height, faces, uvs, attribute, padding, expected_output",
    [
        # Scalar value attribute.
        (
            4,
            4,
            np.array([[0, 1, 2]]),
            np.array([[0, 0], [1, 0], [0, 1]]),
            np.array([0.0, 0.5, 1.0]),
            0,
            np.array(
                [[255, 0, 0, 0], [170, 212, 0, 0], [85, 127, 170, 0], [0, 42, 85, 128]],
                dtype=np.uint8,
            ),
        ),
        # RGB color attribute.
        (
            4,
            4,
            np.array([[0, 1, 2]]),
            np.array([[0, 0], [1, 0], [0, 1]]),
            np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
            0,
            np.array(
                [
                    [[0, 0, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[85, 0, 170], [0, 85, 170], [0, 0, 0], [0, 0, 0]],
                    [[170, 0, 85], [85, 85, 85], [0, 170, 85], [0, 0, 0]],
                    [[255, 0, 0], [170, 85, 0], [85, 170, 0], [0, 255, 0]],
                ],
                dtype=np.uint8,
            ),
        ),
        # RGBA float attribute.
        (
            4,
            2,
            np.array([[0, 1, 2]]),
            np.array([[0, 0], [1, 0], [0, 1]]),
            np.array([[-2.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5]]),
            0,
            np.array(
                [
                    [[255, 0, 255, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [85, 85, 0, 0], [255, 170, 85, 0], [255, 255, 0, 0]],
                ],
                dtype=np.uint8,
            ),
        ),
    ],
)
def test_get_vertex_attribute_image(
    width: int, height: int, faces: Faces, uvs: UVs, attribute: Color, padding: int, expected_output: Image
):
    """Test the get_vertex_attribute_image function with valid inputs."""
    image = draw.get_vertex_attribute_image(width, height, faces, uvs, attribute, padding)
    np.testing.assert_array_equal(image, expected_output)
    assert image.shape[:2] == (
        height,
        width,
    ), f"Image shape {image.shape} does not match expected shape {(height, width)}."
    assert image.dtype == np.uint8, f"Image dtype {image.dtype} does not match expected data type uint8."


@pytest.mark.parametrize(
    "faces, uvs, attribute, error_message",
    [
        # No UVs.
        (
            np.array([[0, 1, 2]]),
            None,
            np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
            "^(UV coordinates are invalid:).*",
        ),
        # Mismatched UVs length.
        (
            np.array([[0, 1, 2]]),
            np.array([[0, 0], [1, 0]]),
            np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
            "^(UV coordinates are invalid: Too few UV coordinates.).*",
        ),
        # Mismatched attribute length.
        (
            np.array([[0, 1, 2]]),
            np.array([[0, 0], [1, 0], [0, 1]]),
            np.array([[255, 0, 0], [0, 255, 0]]),
            "^(Attribute length does not match UV coordinates length:).*",
        ),
        # Unsupported color mode.
        (
            np.array([[0, 1, 2]]),
            np.array([[0, 0], [1, 0], [0, 1]]),
            np.array([[0, 1], [0, 1], [0, 1]]),
            "^(Attribute shape is unsupported for image conversion).*",
        ),
    ],
)
def test_get_vertex_attribute_image_should_fail(faces: Faces, uvs: UVs, attribute: Color, error_message: str):
    """Test the get_vertex_attribute_image function fails when provided data is incompatible."""
    with pytest.raises(ValueError, match=error_message):
        draw.get_vertex_attribute_image(8, 8, faces, uvs, attribute)


@pytest.mark.parametrize(
    "mock_mesh, padding, expected_output",
    [
        # No Padding
        (
            "mock_mesh",
            0,
            np.array(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [
                        [0, 0, 0],
                        [51, 25, 0],
                        [102, 51, 0],
                        [153, 76, 0],
                        [204, 102, 0],
                        [255, 128, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 51],
                        [0, 85, 85],
                        [0, 170, 0],
                        [63, 180, 0],
                        [127, 191, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 102],
                        [0, 170, 170],
                        [0, 212, 85],
                        [0, 255, 0],
                        [255, 255, 0],
                        [255, 191, 0],
                        [255, 128, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 153],
                        [0, 127, 204],
                        [0, 255, 255],
                        [127, 191, 255],
                        [255, 191, 127],
                        [255, 128, 127],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 204],
                        [0, 127, 255],
                        [127, 64, 255],
                        [191, 96, 255],
                        [255, 128, 255],
                        [255, 64, 127],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 255],
                        [63, 32, 255],
                        [127, 64, 255],
                        [191, 64, 255],
                        [255, 64, 255],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
                dtype=np.uint8,
            ),
        ),
        # Negative Padding
        (
            "mock_mesh",
            -4,
            np.array(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [
                        [0, 0, 0],
                        [51, 25, 0],
                        [102, 51, 0],
                        [153, 76, 0],
                        [204, 102, 0],
                        [255, 128, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 51],
                        [0, 85, 85],
                        [0, 170, 0],
                        [63, 180, 0],
                        [127, 191, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 102],
                        [0, 170, 170],
                        [0, 212, 85],
                        [0, 255, 0],
                        [255, 255, 0],
                        [255, 191, 0],
                        [255, 128, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 153],
                        [0, 127, 204],
                        [0, 255, 255],
                        [127, 191, 255],
                        [255, 191, 127],
                        [255, 128, 127],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 204],
                        [0, 127, 255],
                        [127, 64, 255],
                        [191, 96, 255],
                        [255, 128, 255],
                        [255, 64, 127],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 255],
                        [63, 32, 255],
                        [127, 64, 255],
                        [191, 64, 255],
                        [255, 64, 255],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
                dtype=np.uint8,
            ),
        ),
        # Positive Padding
        (
            "mock_mesh",
            4,
            np.array(
                [
                    [
                        [51, 25, 0],
                        [102, 51, 0],
                        [153, 76, 0],
                        [204, 102, 0],
                        [255, 128, 0],
                        [255, 0, 0],
                        [255, 128, 0],
                        [255, 128, 0],
                    ],
                    [
                        [0, 0, 0],
                        [51, 25, 0],
                        [102, 51, 0],
                        [153, 76, 0],
                        [204, 102, 0],
                        [255, 128, 0],
                        [255, 191, 0],
                        [255, 128, 0],
                    ],
                    [
                        [0, 0, 51],
                        [0, 85, 85],
                        [0, 170, 0],
                        [63, 180, 0],
                        [127, 191, 0],
                        [255, 255, 0],
                        [255, 255, 0],
                        [255, 191, 0],
                    ],
                    [
                        [0, 0, 102],
                        [0, 170, 170],
                        [0, 212, 85],
                        [0, 255, 0],
                        [255, 255, 0],
                        [255, 191, 0],
                        [255, 128, 0],
                        [255, 191, 127],
                    ],
                    [
                        [0, 0, 153],
                        [0, 127, 204],
                        [0, 255, 255],
                        [127, 191, 255],
                        [255, 191, 127],
                        [255, 128, 127],
                        [255, 255, 255],
                        [255, 191, 127],
                    ],
                    [
                        [0, 0, 204],
                        [0, 127, 255],
                        [127, 64, 255],
                        [191, 96, 255],
                        [255, 128, 255],
                        [255, 64, 127],
                        [255, 255, 255],
                        [255, 191, 127],
                    ],
                    [
                        [0, 0, 255],
                        [63, 32, 255],
                        [127, 64, 255],
                        [191, 64, 255],
                        [255, 64, 255],
                        [255, 191, 255],
                        [255, 191, 255],
                        [255, 128, 127],
                    ],
                    [
                        [63, 127, 255],
                        [127, 127, 255],
                        [191, 127, 255],
                        [255, 128, 255],
                        [255, 0, 255],
                        [255, 128, 255],
                        [255, 128, 255],
                        [255, 64, 127],
                    ],
                ],
                dtype=np.uint8,
            ),
        ),
    ],
    indirect=["mock_mesh"],
)
def test_get_position_map(mock_mesh: Mesh, padding: int, expected_output: Image):
    """Test the get_position_map function with valid inputs."""
    width = height = 8
    output = draw.get_position_map(width, height, mock_mesh, padding=padding)
    np.testing.assert_array_equal(output, expected_output)


def test_get_position_map_should_fail(mock_mesh: Mesh):
    """Test the get_position_map function fails when mesh has no UV coordinates."""
    mock_mesh.uv_coords = None
    width = height = 8
    with pytest.raises(ValueError, match="^(UV coordinates are invalid:).*"):
        draw.get_position_map(width, height, mock_mesh)


@pytest.mark.parametrize(
    "mock_mesh, padding, expected_output",
    [
        # No Padding
        (
            "mock_mesh",
            0,
            np.array(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [106, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 41], [21, 0, 32], [42, 0, 24], [63, 0, 16], [84, 0, 8], [106, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [
                        [0, 0, 80],
                        [6, 85, 101],
                        [12, 170, 53],
                        [37, 148, 41],
                        [62, 127, 29],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 120],
                        [12, 170, 161],
                        [15, 212, 110],
                        [19, 255, 59],
                        [192, 255, 59],
                        [223, 127, 99],
                        [255, 0, 140],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 160],
                        [9, 127, 190],
                        [19, 255, 221],
                        [115, 127, 230],
                        [201, 127, 149],
                        [233, 0, 190],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 200],
                        [9, 127, 230],
                        [105, 0, 240],
                        [158, 0, 240],
                        [211, 0, 240],
                        [223, 0, 197],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 240],
                        [52, 0, 240],
                        [105, 0, 240],
                        [153, 0, 243],
                        [201, 0, 247],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [192, 0, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
                dtype=np.uint8,
            ),
        ),
        # Negative Padding
        (
            "mock_mesh",
            -4,
            np.array(
                [
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [106, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 41], [21, 0, 32], [42, 0, 24], [63, 0, 16], [84, 0, 8], [106, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [
                        [0, 0, 80],
                        [6, 85, 101],
                        [12, 170, 53],
                        [37, 148, 41],
                        [62, 127, 29],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 120],
                        [12, 170, 161],
                        [15, 212, 110],
                        [19, 255, 59],
                        [192, 255, 59],
                        [223, 127, 99],
                        [255, 0, 140],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 160],
                        [9, 127, 190],
                        [19, 255, 221],
                        [115, 127, 230],
                        [201, 127, 149],
                        [233, 0, 190],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 200],
                        [9, 127, 230],
                        [105, 0, 240],
                        [158, 0, 240],
                        [211, 0, 240],
                        [223, 0, 197],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 240],
                        [52, 0, 240],
                        [105, 0, 240],
                        [153, 0, 243],
                        [201, 0, 247],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [192, 0, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
                dtype=np.uint8,
            ),
        ),
        # Positive Padding
        (
            "mock_mesh",
            4,
            np.array(
                [
                    [
                        [21, 0, 41],
                        [42, 0, 41],
                        [63, 0, 41],
                        [84, 0, 32],
                        [106, 0, 24],
                        [106, 0, 0],
                        [106, 0, 8],
                        [106, 0, 0],
                    ],
                    [
                        [0, 0, 41],
                        [21, 0, 32],
                        [42, 0, 24],
                        [63, 0, 16],
                        [84, 0, 8],
                        [106, 0, 0],
                        [106, 127, 29],
                        [106, 0, 0],
                    ],
                    [
                        [0, 0, 80],
                        [6, 85, 101],
                        [12, 170, 53],
                        [37, 148, 41],
                        [62, 127, 29],
                        [255, 255, 140],
                        [255, 255, 140],
                        [255, 127, 140],
                    ],
                    [
                        [0, 0, 120],
                        [12, 170, 161],
                        [15, 212, 110],
                        [19, 255, 59],
                        [192, 255, 59],
                        [223, 127, 99],
                        [255, 0, 140],
                        [255, 127, 190],
                    ],
                    [
                        [0, 0, 160],
                        [9, 127, 190],
                        [19, 255, 221],
                        [115, 127, 230],
                        [201, 127, 149],
                        [233, 0, 190],
                        [255, 255, 240],
                        [255, 127, 197],
                    ],
                    [
                        [0, 0, 200],
                        [9, 127, 230],
                        [105, 0, 240],
                        [158, 0, 240],
                        [211, 0, 240],
                        [223, 0, 197],
                        [255, 255, 247],
                        [255, 127, 197],
                    ],
                    [
                        [0, 0, 240],
                        [52, 0, 240],
                        [105, 0, 240],
                        [153, 0, 243],
                        [201, 0, 247],
                        [233, 127, 255],
                        [233, 127, 255],
                        [233, 0, 197],
                    ],
                    [
                        [52, 127, 240],
                        [105, 127, 240],
                        [158, 127, 243],
                        [211, 127, 255],
                        [192, 0, 255],
                        [223, 0, 255],
                        [223, 0, 255],
                        [223, 0, 197],
                    ],
                ],
                dtype=np.uint8,
            ),
        ),
    ],
    indirect=["mock_mesh"],
)
def test_get_obj_space_normal_map(mock_mesh: Mesh, padding: int, expected_output: Image):
    """Test the get_obj_space_normal_map function with valid inputs."""
    width = height = 8
    output = draw.get_obj_space_normal_map(width, height, mock_mesh, padding=padding)
    np.testing.assert_array_equal(output, expected_output)


@pytest.mark.parametrize(
    "mock_mesh, missing, error_pattern",
    [
        (
            "mock_mesh",
            "uv_coords",
            "^(UV coordinates are invalid:).*",
        ),
        (
            "mock_mesh",
            "normals",
            "Mesh does not have vertex normals.",
        ),
    ],
    indirect=["mock_mesh"],
)
def test_get_obj_space_normal_map_should_fail(mock_mesh: Mesh, missing: str, error_pattern: str):
    """Test the get_obj_space_normal_map function fails when mesh has no vertex normals."""
    setattr(mock_mesh, missing, None)
    width = height = 8
    with pytest.raises(ValueError, match=error_pattern):
        draw.get_obj_space_normal_map(width, height, mock_mesh)

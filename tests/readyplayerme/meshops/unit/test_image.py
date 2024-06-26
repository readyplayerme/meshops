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
    ids=[
        "Basic blending with a uniform mask",
        "Blending with different mask values",
        "Mask with all zeros (full image_1)",
        "Mask with all ones (full image_2)",
        "Non uniform mask values:",
        "Full RGB Image",
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
    ids=["Shape mismatch", "Invalid input type", "Empty mask"],
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
    ids=[
        "All NaNs",
        "No NaNs",
        "Single Element",
        "Single NaN",
        "Interpolation with NaNs in the middle",
        "Interpolation with multiple NaNs",
        "NaN at the beginning",
        "NaN at the end",
        "NaNs at both ends",
    ],
)
def test_interpolate_segment(input_segment: Image, expected_output: Image):
    """Test the interpolate_segment function with various input scenarios."""
    output = draw.interpolate_segment(input_segment)
    np.testing.assert_array_equal(output, expected_output)


@pytest.mark.parametrize(
    "image, edges, image_coords, colors, interpolate_func, expected_output",
    [
        # Empty Edges
        (
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.array([]),
            np.array([[0, 0], [1, 1]]),
            np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
            lambda color0, color1, steps: np.array([[100, 100, 100]] * steps, dtype=np.uint8),  # noqa: ARG005
            np.zeros((5, 5, 3), dtype=np.uint8),
        ),
        # RGBA image
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
        # Grayscale image, grayscale 2D colors (2,1)
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
        # Grayscale image, grayscale 1D colors (2,)
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
        # Non-Existent Edge Points
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
        # Zero Length Lines
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
    ids=[
        "Empty Edges",
        "RGBA image",
        "Grayscale image, grayscale 2D colors (2,1)",
        "Grayscale image, grayscale 1D colors (2,)",
        "Non-Existent Edge Points",
        "Zero Length Lines",
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
    ids=[
        "Interpolate NaNs in-between valid values",
        "Horizontal interpolation with multiple columns",
        "Multiple NaNs in-between",
        "No NaNs in-between",
        "NaNs only at the edges should remain as NaNs",
        "Single NaN in-between",
        "All NaNs except edges",
        "Single-Row Array with NaN in-between",
        "Empty Arrays",
        "Single Element Arrays",
        "Arrays with No NaN Values",
        "All NaN Arrays",
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
        # Multiple columns with nan edges Grayscale
        (np.array([[1], [2], [np.nan]]), np.array([[1], [2], [2]])),
        # Multiple columns with nan edges RGB
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
    ids=[
        "Basic vertical interpolation single columns",
        "Basic vertical interpolation multiple columns",
        "Edge cases",
        "Multiple columns with nan edges Grayscale",
        "Multiple columns with nan edges RGB",
        "Multiple consecutive NaNs",
        "No NaNs",
        "All NaNs",
        "Single-column Array",
        "Empty Arrays",
        "Single Element Arrays",
        "Arrays with No NaN Values",
        "All NaN Arrays",
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
    ids=["RGB", "RGBA", "Grayscale"],
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
    ids=["Zero width", "Zero height", "Negative width", "Negative height", "Float width", "Float height"],
)
def test_create_nan_image_should_fail(width: int, height: int, error: type[Exception]):
    """Test the create_nan_image function with invalid inputs."""
    with pytest.raises(error):
        draw.create_nan_image(width, height)


@pytest.mark.parametrize(
    "input_array, expected_output",
    [
        # No NaNs or Infinities, uint8
        (np.array([[100, 150], [200, 250]], dtype=np.uint8), np.array([[100, 150], [200, 250]], dtype=np.uint8)),
        # No NaNs or Infinities, float
        (np.array([[1.0, 0.0], [0.5, 0.25]], dtype=np.float32), np.array([[1.0, 0.0], [0.5, 0.25]], dtype=np.float32)),
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
        # Mix of finite, NaNs and Infinities
        (
            np.array([[np.nan, 150, -np.inf], [np.inf, 75, np.nan]], dtype=np.float32),
            np.array([[0, 150, 0], [255, 75, 0]], dtype=np.float32),
        ),
        # Values Exceeding the Range [0, 255]
        (np.array([[300, -100], [500, 600]], dtype=np.float32), np.array([[255, 0], [255, 255]], dtype=np.float32)),
    ],
    ids=[
        "Finites only, uint8",
        "Finites only, float",
        "NaNs",
        "Positive and Negative Infinities",
        "Mix of NaNs and Infinities",
        "Mix of finite, NaNs and Infinities",
        "Values Exceeding the Range [0, 255]",
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
    ids=["Basic Functionality"],
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
    ids=["Basic Functionality", "No Edges", "Single Pixel Image"],
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
    ids=[
        "Out of Bounds Edges",
        "Zero Dimensions",
        "Mismatched Array Sizes",
        "Mismatched Color Modes (Grayscale+RGB)",
        "Mismatched Color Modes (RGB+Grayscale)",
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
    ids=["Grayscale image", "RGB image", "RGBA image"],
    indirect=["mock_mesh"],
)
def test_blend_uv_seams(mock_mesh: Mesh, image: Image, expected_output: Image):
    """Test the blend_uv_seams function with valid inputs."""
    output = draw.blend_uv_seams(mock_mesh, image)
    np.testing.assert_array_equal(output, expected_output)


def test_blend_uv_seams_should_fail(mock_mesh: Mesh):
    """Test the blend_uv_seams function fails when mesh has no UV coordinates."""
    mock_mesh.uv_coords = None
    image = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="^(UV coordinates are invalid:).*"):
        draw.blend_uv_seams(mock_mesh, image)


@pytest.mark.parametrize(
    "width, height, faces, uvs, attribute, padding, normalize_per_channel, expected_output",
    [
        # TODO: Add more tests: UV, 0-dim, normalize_per_channel
        # Scalar value attribute.
        (
            4,
            4,
            np.array([[0, 1, 2]]),
            np.array([[0, 0], [1, 0], [0, 1]]),
            np.array([0.0, 0.5, 1.0]),
            0,
            False,
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
            False,
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
            False,
            np.array(
                [
                    [[128, 128, 191, 159], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 128, 128, 159], [42, 149, 128, 159], [128, 170, 149, 159], [128, 191, 128, 159]],
                ],
                dtype=np.uint8,
            ),
        ),
    ],
    ids=["Scalar value attribute", "RGB color attribute", "RGBA float attribute"],
)
def test_get_vertex_attribute_image(
    width: int,
    height: int,
    faces: Faces,
    uvs: UVs,
    attribute: Color,
    padding: int,
    normalize_per_channel: bool,  # noqa: FBT001
    expected_output: Image,
):
    """Test the get_vertex_attribute_image function with valid inputs."""
    image = draw.get_vertex_attribute_image(
        width, height, faces, uvs, attribute, padding, normalize_per_channel=normalize_per_channel
    )
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
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]),
            "^(Attribute shape is unsupported for image conversion).*",
        ),
    ],
    ids=["No UVs", "Mismatched UVs length", "Mismatched attribute length", "Unsupported color mode"],
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
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [167, 74, 78], [0, 0, 0], [0, 0, 0]],
                    [
                        [59, 74, 78],
                        [80, 84, 78],
                        [102, 95, 78],
                        [123, 106, 78],
                        [145, 117, 78],
                        [167, 128, 78],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [59, 74, 99],
                        [59, 109, 114],
                        [59, 145, 78],
                        [86, 149, 78],
                        [113, 154, 78],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [59, 74, 121],
                        [59, 145, 150],
                        [59, 163, 114],
                        [59, 181, 78],
                        [167, 181, 78],
                        [167, 154, 78],
                        [167, 128, 78],
                        [0, 0, 0],
                    ],
                    [
                        [59, 74, 142],
                        [59, 127, 164],
                        [59, 181, 186],
                        [113, 154, 186],
                        [167, 154, 132],
                        [167, 128, 132],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [59, 74, 164],
                        [59, 127, 186],
                        [113, 101, 186],
                        [140, 114, 186],
                        [167, 128, 186],
                        [167, 101, 132],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [59, 74, 186],
                        [86, 87, 186],
                        [113, 101, 186],
                        [140, 101, 186],
                        [167, 101, 186],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [167, 74, 186], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
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
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [167, 74, 78], [0, 0, 0], [0, 0, 0]],
                    [
                        [59, 74, 78],
                        [80, 84, 78],
                        [102, 95, 78],
                        [123, 106, 78],
                        [145, 117, 78],
                        [167, 128, 78],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [59, 74, 99],
                        [59, 109, 114],
                        [59, 145, 78],
                        [86, 149, 78],
                        [113, 154, 78],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [59, 74, 121],
                        [59, 145, 150],
                        [59, 163, 114],
                        [59, 181, 78],
                        [167, 181, 78],
                        [167, 154, 78],
                        [167, 128, 78],
                        [0, 0, 0],
                    ],
                    [
                        [59, 74, 142],
                        [59, 127, 164],
                        [59, 181, 186],
                        [113, 154, 186],
                        [167, 154, 132],
                        [167, 128, 132],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [59, 74, 164],
                        [59, 127, 186],
                        [113, 101, 186],
                        [140, 114, 186],
                        [167, 128, 186],
                        [167, 101, 132],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [59, 74, 186],
                        [86, 87, 186],
                        [113, 101, 186],
                        [140, 101, 186],
                        [167, 101, 186],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [167, 74, 186], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
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
                        [80, 84, 78],
                        [102, 95, 78],
                        [123, 106, 78],
                        [145, 117, 78],
                        [167, 128, 78],
                        [167, 74, 78],
                        [167, 128, 78],
                        [167, 128, 78],
                    ],
                    [
                        [59, 74, 78],
                        [80, 84, 78],
                        [102, 95, 78],
                        [123, 106, 78],
                        [145, 117, 78],
                        [167, 128, 78],
                        [167, 154, 78],
                        [167, 128, 78],
                    ],
                    [
                        [59, 74, 99],
                        [59, 109, 114],
                        [59, 145, 78],
                        [86, 149, 78],
                        [113, 154, 78],
                        [167, 181, 78],
                        [167, 181, 78],
                        [167, 154, 78],
                    ],
                    [
                        [59, 74, 121],
                        [59, 145, 150],
                        [59, 163, 114],
                        [59, 181, 78],
                        [167, 181, 78],
                        [167, 154, 78],
                        [167, 128, 78],
                        [167, 154, 132],
                    ],
                    [
                        [59, 74, 142],
                        [59, 127, 164],
                        [59, 181, 186],
                        [113, 154, 186],
                        [167, 154, 132],
                        [167, 128, 132],
                        [167, 181, 186],
                        [167, 154, 132],
                    ],
                    [
                        [59, 74, 164],
                        [59, 127, 186],
                        [113, 101, 186],
                        [140, 114, 186],
                        [167, 128, 186],
                        [167, 101, 132],
                        [167, 181, 186],
                        [167, 154, 132],
                    ],
                    [
                        [59, 74, 186],
                        [86, 87, 186],
                        [113, 101, 186],
                        [140, 101, 186],
                        [167, 101, 186],
                        [167, 154, 186],
                        [167, 154, 186],
                        [167, 128, 132],
                    ],
                    [
                        [86, 127, 186],
                        [113, 127, 186],
                        [140, 127, 186],
                        [167, 128, 186],
                        [167, 74, 186],
                        [167, 128, 186],
                        [167, 128, 186],
                        [167, 101, 132],
                    ],
                ],
                dtype=np.uint8,
            ),
        ),
        # TODO: tests for center and uniform_scale
    ],
    ids=["No Padding", "Negative Padding", "Positive Padding"],
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
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [128, 128, 0], [0, 0, 0], [0, 0, 0]],
                    [
                        [37, 128, 37],
                        [55, 128, 29],
                        [73, 128, 22],
                        [91, 128, 14],
                        [109, 128, 7],
                        [128, 128, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [37, 128, 73],
                        [42, 152, 91],
                        [48, 176, 48],
                        [69, 170, 37],
                        [91, 164, 27],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [37, 128, 109],
                        [48, 176, 146],
                        [51, 188, 100],
                        [54, 201, 54],
                        [201, 201, 54],
                        [228, 164, 91],
                        [255, 128, 128],
                        [0, 0, 0],
                    ],
                    [
                        [37, 128, 145],
                        [45, 164, 173],
                        [54, 201, 201],
                        [136, 164, 209],
                        [209, 164, 136],
                        [236, 128, 173],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [37, 128, 181],
                        [45, 164, 209],
                        [127, 128, 218],
                        [172, 128, 218],
                        [218, 128, 218],
                        [228, 128, 180],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [37, 128, 218],
                        [82, 128, 218],
                        [127, 128, 218],
                        [168, 128, 221],
                        [209, 128, 225],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [201, 128, 232], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
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
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [128, 128, 0], [0, 0, 0], [0, 0, 0]],
                    [
                        [37, 128, 37],
                        [55, 128, 29],
                        [73, 128, 22],
                        [91, 128, 14],
                        [109, 128, 7],
                        [128, 128, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [37, 128, 73],
                        [42, 152, 91],
                        [48, 176, 48],
                        [69, 170, 37],
                        [91, 164, 27],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [37, 128, 109],
                        [48, 176, 146],
                        [51, 188, 100],
                        [54, 201, 54],
                        [201, 201, 54],
                        [228, 164, 91],
                        [255, 128, 128],
                        [0, 0, 0],
                    ],
                    [
                        [37, 128, 145],
                        [45, 164, 173],
                        [54, 201, 201],
                        [136, 164, 209],
                        [209, 164, 136],
                        [236, 128, 173],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [37, 128, 181],
                        [45, 164, 209],
                        [127, 128, 218],
                        [172, 128, 218],
                        [218, 128, 218],
                        [228, 128, 180],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [37, 128, 218],
                        [82, 128, 218],
                        [127, 128, 218],
                        [168, 128, 221],
                        [209, 128, 225],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [201, 128, 232], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
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
                        [55, 128, 37],
                        [73, 128, 37],
                        [91, 128, 37],
                        [109, 128, 29],
                        [128, 128, 22],
                        [128, 128, 0],
                        [128, 128, 7],
                        [128, 128, 0],
                    ],
                    [
                        [37, 128, 37],
                        [55, 128, 29],
                        [73, 128, 22],
                        [91, 128, 14],
                        [109, 128, 7],
                        [128, 128, 0],
                        [128, 164, 27],
                        [128, 128, 0],
                    ],
                    [
                        [37, 128, 73],
                        [42, 152, 91],
                        [48, 176, 48],
                        [69, 170, 37],
                        [91, 164, 27],
                        [255, 201, 128],
                        [255, 201, 128],
                        [255, 164, 128],
                    ],
                    [
                        [37, 128, 109],
                        [48, 176, 146],
                        [51, 188, 100],
                        [54, 201, 54],
                        [201, 201, 54],
                        [228, 164, 91],
                        [255, 128, 128],
                        [255, 164, 173],
                    ],
                    [
                        [37, 128, 145],
                        [45, 164, 173],
                        [54, 201, 201],
                        [136, 164, 209],
                        [209, 164, 136],
                        [236, 128, 173],
                        [255, 201, 218],
                        [255, 164, 180],
                    ],
                    [
                        [37, 128, 181],
                        [45, 164, 209],
                        [127, 128, 218],
                        [172, 128, 218],
                        [218, 128, 218],
                        [228, 128, 180],
                        [255, 201, 225],
                        [255, 164, 180],
                    ],
                    [
                        [37, 128, 218],
                        [82, 128, 218],
                        [127, 128, 218],
                        [168, 128, 221],
                        [209, 128, 225],
                        [236, 164, 232],
                        [236, 164, 232],
                        [236, 128, 180],
                    ],
                    [
                        [82, 164, 218],
                        [127, 164, 218],
                        [172, 164, 221],
                        [218, 164, 232],
                        [201, 128, 232],
                        [228, 128, 232],
                        [228, 128, 232],
                        [228, 128, 180],
                    ],
                ],
                dtype=np.uint8,
            ),
        ),
    ],
    ids=["No Padding", "Negative Padding", "Positive Padding"],
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
    ids=["No UVs", "No Normals"],
    indirect=["mock_mesh"],
)
def test_get_obj_space_normal_map_should_fail(mock_mesh: Mesh, missing: str, error_pattern: str):
    """Test the get_obj_space_normal_map function fails when mesh has no vertex normals."""
    setattr(mock_mesh, missing, None)
    width = height = 8
    with pytest.raises(ValueError, match=error_pattern):
        draw.get_obj_space_normal_map(width, height, mock_mesh)

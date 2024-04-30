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
    ids=["Grayscale", "RGB", "RGBA"],
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
    ids=["Invalid channel count", "1D array", "4D array"],
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
    ids=["Grayscale 1D", "Grayscale 2D", "RGB", "RGBA"],
)
def test_get_color_mode(color_array, expected_mode):
    """Test the get_color_mode function with valid inputs."""
    assert draw.get_color_mode(color_array) == expected_mode


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
    ids=["Invalid channel count", "3D array", "0-dimensional array"],
)
def test_get_color_mode_should_fail(color_array):
    """Test the get_color_mode function with invalid inputs."""
    with pytest.raises(ValueError):
        draw.get_color_mode(color_array)


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
    ids=[
        "Simple groups and distinct colors",
        "Single group",
        "Groups of 1 element",
        "Empty colors array",
        "Empty groups",
        "Empty colors and groups",
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
    ids=["Out-of-bounds indices", "Negative index"],
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
    ids=[
        "Basic interpolation from red to blue",
        "Interpolation from RGBA red to RGBA blue (with alpha channel)",
        "Interpolation in grayscale",
        "Interpolation with more steps",
        "Interpolation with a single step",
        "Interpolation with two steps",
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
    ids=["Mismatched shapes", "Invalid number of steps (less than 1)", "Invalid type for start or end color"],
)
def test_interpolate_values_should_fail(start_color, end_color, num_steps, expected_exception):
    """Test the vectorized_interpolate function with invalid input scenarios."""
    with pytest.raises(expected_exception):
        draw.interpolate_values(start_color, end_color, num_steps)


@pytest.mark.parametrize(
    "attribute, per_channel_normalization, expected",
    [
        # Test with single scalar attribute.
        (np.array([-5]), False, np.array([0], dtype=np.uint8)),
        # Test with uint8 scalar attribute.
        (
            np.array([128, 64, 192], dtype=np.uint8),
            False,
            np.array([128, 64, 192], dtype=np.uint8),
        ),
        # Test with uint8 "1.5D" attribute.
        (
            np.array([[128], [64], [192]], dtype=np.uint8),
            False,
            np.array([128, 64, 192], dtype=np.uint8),
        ),
        # Test with uint8 2D attribute.
        (
            np.array([[128, 192], [64, 32]], dtype=np.uint8),
            False,
            np.array([[128, 192, 0], [64, 32, 0]], dtype=np.uint8),
        ),
        # Test with uint8 RGBA attribute.
        (
            np.array([[128, 64, 192, 32], [100, 150, 200, 50]], dtype=np.uint8),
            False,
            np.array([[128, 64, 192, 32], [100, 150, 200, 50]], dtype=np.uint8),
        ),
        # Test with int32 RGB attribute and per_channel_normalization=True
        (
            np.array([[200, 0, 500], [-200, 0, 250], [100, 0, 0]], dtype=np.int32),
            True,
            np.array([[255, 0, 255], [0, 0, 127], [191, 0, 0]], dtype=np.uint8),
        ),
        # Test with float32 attribute and per_channel_normalization=True
        (
            np.array([[-2.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5]]),
            True,
            np.array([[0, 0, 0, 0], [255, 255, 0, 0], [255, 0, 255, 0]], dtype=np.uint8),
        ),
        # Test with float32 attribute and per_channel_normalization=False
        (
            np.array([[-2.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5]]),
            False,
            np.array([[0, 170, 170, 212], [170, 255, 170, 212], [170, 170, 255, 212]], dtype=np.uint8),
        ),
    ],
    ids=[
        "single scalar",
        "uint8 scalars",
        "uint8 1.5D",
        "uint8 2D",
        "uint8 RGBA",
        "int32 RGB",
        "float32 RGBA per channel",
        "float32 RGBA global",
    ],
)
def test_attribute_to_color(attribute, per_channel_normalization, expected):
    """Test the attribute_to_color function with valid inputs."""
    result = draw.attribute_to_color(attribute, normalize_per_channel=per_channel_normalization)
    assert result.dtype == np.uint8
    assert len(result) == len(attribute)  # The number of colors should match the number of attributes.
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "attribute",
    [
        # Test with empty attribute
        np.array([]),
        # Test with 0-dimensional attribute
        np.array(5),
    ],
    ids=["empty", "0-dimensional attribute"],
)
def test_attribute_to_color_should_fail(attribute):
    """Test the attribute_to_color function with invalid inputs."""
    with pytest.raises(ValueError):
        draw.attribute_to_color(attribute)

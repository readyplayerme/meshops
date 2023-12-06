"""Unit tests for the image module."""

import numpy as np
import pytest

import readyplayerme.meshops.image as img
from readyplayerme.meshops.types import ColorMode


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
def test_blend_images(image_1, image_2, mask, expected_output):
    """Test the blend_images function with various input scenarios."""
    output = img.blend_images(image_1, image_2, mask)
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
def test_blend_images_should_fail(image_1, image_2, mask, expected_exception):
    """Test the blend_images function with invalid input scenarios."""
    with pytest.raises(expected_exception):
        img.blend_images(image_1, image_2, mask)


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
    assert img.get_image_color_mode(image) == expected_mode


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
        img.get_image_color_mode(image)


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
    assert img.get_color_array_color_mode(color_array) == expected_mode


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
        img.get_color_array_color_mode(color_array)

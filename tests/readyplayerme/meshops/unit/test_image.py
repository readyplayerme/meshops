"""Unit tests for the image module."""

import numpy as np
import pytest

from readyplayerme.meshops import draw


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
def test_blend_images_should_fail(image_1, image_2, mask, expected_exception):
    """Test the blend_images function with invalid input scenarios."""
    with pytest.raises(expected_exception):
        draw.blend_images(image_1, image_2, mask)

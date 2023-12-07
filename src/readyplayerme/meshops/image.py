from readyplayerme.meshops.types import Color, ColorMode
from readyplayerme.meshops.types import Image as IMG_type


def get_image_color_mode(image: IMG_type) -> ColorMode:
    """
    Determine the color mode of an image.

    :param image: An image array.
    :return ColorMode:: Enum indicating the color mode of the image (GRAYSCALE, RGB, or RGBA).
    """
    try:
        n_channels = image.shape[-1]
    except IndexError as error:
        error_msg = "Image has invalid shape: zero dimensions."
        raise ValueError(error_msg) from error
    match image.ndim, n_channels:
        case 2, _:
            return ColorMode.GRAYSCALE
        case 3, 3:
            return ColorMode.RGB
        case 3, 4:
            return ColorMode.RGBA
        case _:
            error_msg = "Invalid color mode for an image."
            raise ValueError(error_msg)


def get_color_array_color_mode(color_array: Color) -> ColorMode:
    """
    Determine the color mode of a color array.

    :param color_array: An array representing colors.
    :return ColorMode: Enum indicating the color mode of the color array (GRAYSCALE, RGB, or RGBA).
    """
    try:
        n_channels = color_array.shape[-1]
    except IndexError as error:
        error_msg = "Color has invalid shape: zero dimensions."
        raise ValueError(error_msg) from error
    match color_array.ndim, n_channels:
        case 1, _:
            return ColorMode.GRAYSCALE
        case 2, 1:
            return ColorMode.GRAYSCALE
        case 2, 3:
            return ColorMode.RGB
        case 2, 4:
            return ColorMode.RGBA
        case _:
            msg = "Invalid dimensions for a color array."
            raise ValueError(msg)


def blend_images(image1: IMG_type, image2: IMG_type, mask: IMG_type) -> IMG_type:
    """
    Blend two images using a mask.

    This function performs a blending operation on two images using a mask. The mask determines the blending ratio
    at each pixel. The blending is done via a vectorized operation for efficiency.

    :param image1: The first image to blend, as a NumPy array.
    :param image2: The second image to blend, as a NumPy array. Must be the same shape as the first image.
    :param mask: The blending mask, as a NumPy array. Must be the same shape as the images.
    :return: The blended image.
    """
    try:
        # Check if the error is due to shape mismatch
        if not (image1.shape == image2.shape):
            msg = "image1 and image2 must have the same shape."
            raise ValueError(msg)
        # Mask is reshaped to be able to perform the blending
        expected_dimensions = 2
        if mask.ndim == expected_dimensions and image1.ndim == expected_dimensions + 1:
            mask = mask[:, :, None]
        # Perform the blending operation using vectorized NumPy operations
        blended_image = (1 - mask) * image1 + mask * image2
        return blended_image
    except AttributeError as error:
        # Re-raise the original exception if it's not a shape mismatch
        msg = "Could not blend the two images with the given mask"
        raise AttributeError(msg) from error

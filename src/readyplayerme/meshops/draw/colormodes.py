"""Get, set, and convert color modes."""
from readyplayerme.meshops.types import Color, ColorMode, Image


def get_image_color_mode(image: Image) -> ColorMode:
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

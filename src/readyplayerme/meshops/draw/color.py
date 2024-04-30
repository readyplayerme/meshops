"""Functions to deal with colors and color modes."""

from typing import Any

import numpy as np
import numpy.typing as npt
import skimage

from readyplayerme.meshops.types import Color, ColorMode, Image, IndexGroups


def get_image_color_mode(image: Image) -> ColorMode:
    """
    Determine the color mode of an image.

    If the image only has 2 dimensions, it is assumed to be grayscale.
    If the image has 3 dimensions, the last dimension is checked for the number of channels.
    With 3 channels, the image is assumed to be RGB. With 4 channels, the image is assumed to be RGBA.
    Other dimensions are not supported.

    :param image: An image.
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


def get_color_mode(color: Color) -> ColorMode:
    """
    Determine the color mode of a color array.

    :param color: An array representing colors.
    :return ColorMode: Enum indicating the color mode of the color array (GRAYSCALE, RGB, or RGBA).
    """
    try:
        n_channels = color.shape[-1]
    except IndexError as error:
        error_msg = "Color has invalid shape: zero dimensions."
        raise ValueError(error_msg) from error
    match color.ndim, n_channels:
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


def blend_colors(colors: Color, index_groups: IndexGroups) -> Color:
    """Blend colors according to the given grouped indices.

    Colors at indices in the same group are blended into having the same color.

    :param index_groups: Groups of indices. Indices must be within the bounds of the colors array.
    :param vertex_colors: Colors.
    :return: Colors with new blended colors at given indices.
    """
    if not len(colors):
        return np.empty_like(colors)

    blended_colors = np.copy(colors)
    if not index_groups:
        return blended_colors

    # Check that the indices are within the bounds of the colors array,
    # so we don't start any operations that would later fail.
    try:
        colors[np.hstack(index_groups)]
    except IndexError as error:
        msg = f"Index in index groups is out of bounds for colors: {error}"
        raise IndexError(msg) from error

    # Blending process
    for group in index_groups:
        if len(group):  # A mean-operation with an empty group would return nan values.
            blended_colors[group] = np.mean(colors[group], axis=0)

    return blended_colors


def interpolate_values(start: Color, end: Color, num_steps: int) -> Color:
    """
    Return an array with interpolated values between start and end.

    The array includes start and end values and is of length num_steps+2, with the first element being start,
    and the last being end, and in between elements are linearly interpolated between these values.

    :param start: The starting value(s) for interpolation.Ex. Colors (G, RGB, RGBA), Normals, etc..
    :param end: The ending value(s) for interpolation.Ex. Colors (G, RGB, RGBA), Normals, etc..
    :param num_steps: The number of interpolation steps.
    :return: An array of interpolated values.
    """
    if start.shape != end.shape:
        msg = "Start and end values must have the same shape."
        raise ValueError(msg)

    if num_steps < 1:
        msg = "Number of steps must be at least 1."
        raise ValueError(msg)

    t = np.arange(num_steps) / max(num_steps - 1, 1)

    if start.size == 1 and start.ndim == 1:
        return start + t * (end - start)
    else:
        return start[None, :] + t[:, None] * (end - start)


def attribute_to_color(attribute: npt.NDArray[Any], *, normalize_per_channel: bool = True) -> Color:
    """Convert an attribute to color values.

    If necessary, normalize it and convert it to uint8.

    :param attribute: The attribute to turn into color values.
    :param normalize_per_channel: Whether to normalize each channel separately or across all channels.
    Only if the attribute is not already uint8.
    :return: The attribute as uint8 color values.
    """
    if attribute.ndim == 0:
        msg = "Attribute has 0 dimensions. Must at least be a scalar (1 dimension)."
        raise ValueError(msg)
    if attribute.size > 1:  # Do not squeeze a scalar, as it will get 0 dimensions.
        attribute = np.squeeze(attribute)
    # If the attribute has 2 values, like UV coordinates, add a third value to make it RGB.
    dim2 = 2
    if attribute.ndim == dim2 and attribute.shape[1] == dim2:
        attribute = np.pad(attribute, ((0, 0), (0, 1)), mode="constant", constant_values=0)
    # A color should not have more than 4 channels, but we don't enforce it here.
    # Normalize the attribute.
    if attribute.dtype != np.uint8:
        axis = 0 if normalize_per_channel else None
        colors = np.nan_to_num(
            (attribute - attribute.min(axis=axis, keepdims=True))
            / (np.ptp(attribute, axis=axis, keepdims=True) + 1e-7)  # Small constant to avoid division by zero.
        )
        colors = skimage.util.img_as_ubyte(colors)
    else:
        colors = attribute

    return colors

from collections.abc import Callable

import numpy as np
import skimage

from readyplayerme.meshops.types import Color, ColorMode, Edges, Image, PixelCoord


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
    return start[None, :] + t[:, None] * (end - start)


def interpolate_segment(segment: Image) -> Image:
    """
    Interpolate NaN values in a 1D numpy array.

    This function linearly interpolates NaN values in the provided array using the nearest non-NaN values.
    Edge NaNs are filled with the nearest valid values.

    :param segment: A 1D NumPy array containing numerical values, where some might be NaN. This 'segment'
    could be a row or a column from a 2D array.
    :return: A 1D NumPy array where NaN values have been replaced by interpolated values
    based on adjacent non-NaN elements.
    """
    nan_mask = np.isnan(segment)

    # Return early if no NaNs are present
    if not np.any(nan_mask):
        return segment

    # Return as-is if all values are NaN
    if nan_mask.all():
        return segment

    x = np.arange(segment.size)
    valid_x = x[~nan_mask]
    valid_segment = segment[~nan_mask]
    # For edge NaNs, use the nearest valid values
    left_value = valid_segment[0] if len(valid_segment) > 0 else np.nan
    right_value = valid_segment[-1] if len(valid_segment) > 0 else np.nan

    return np.interp(x, valid_x, valid_segment, left=left_value, right=right_value)


def lerp_nans_horizontally(image: Image) -> Image:
    """
    Linearly interpolates over NaN values in a 2D array, horizontally.

    This function applies linear interpolation across each row of the array, filling NaN values based on adjacent
    non-NaN elements in the same row. Edge NaNs in a row are filled with the nearest valid values in that row.

    :param image: A 2D NumPy array to interpolate over. Each row of the array is processed separately.
    :return: A 2D NumPy array with NaN values in each row replaced by interpolated values.
    """
    if image.ndim == 1:
        image = image[np.newaxis, :]

    return np.apply_along_axis(interpolate_segment, 1, image)


def lerp_nans_vertically(image: Image) -> Image:
    """Linearly interpolates over NaN values in a 2D array, vertically.

    This function applies linear interpolation across each column of the array, filling NaN values based on adjacent
    non-NaN elements in the same column. Edge NaNs in a column are filled with the nearest valid values in that column.

    :param image: The array to interpolate over.
    :return: The interpolated array.
    """
    if image.ndim == 1:
        image = image[:, np.newaxis]

    return np.apply_along_axis(interpolate_segment, 0, image)


def create_nan_image(width: int, height: int, mode: ColorMode = ColorMode.RGB) -> Image:
    """
    Create an image filled with NaN values.

    :param width: Width of the image in pixels.
    :param height: Height of the image in pixels.
    :param mode: The color mode of the image. Default RGB.
    :return: An RGB image of height x width, filled with NaN values.
    """
    try:
        # Manual check since np.full does not care if is negative
        if width <= 0 or height <= 0:
            msg = "Width and height must be positive integers"
            raise ValueError(msg)

        shape = (height, width) if mode == ColorMode.GRAYSCALE else (height, width, mode.value)
        return np.full(shape, np.nan, dtype=np.float32)
    except ValueError as error:
        msg = "Failed to create NaN image"
        raise ValueError(msg) from error


def draw_lines(
    image: Image,
    edges: Edges,
    image_coords: PixelCoord,
    colors: Color,
    interpolate_func: Callable[[Color, Color, int], Color] = interpolate_values,
) -> Image:
    """
    Draw lines with color interpolation on an image.

    :param image: The image to draw lines on.
    :param edges: List of tuples representing the start and end indices of the edges
    from the image_coords and colors array.
    :param image_coords: Texture coordinates for the edges.
    :param colors: Array of colors.
    :param interpolate_func: Function to interpolate colors.
    :return: Image with interpolated lines.
    """
    for edge in edges:
        try:
            color0, color1 = colors[edge].astype(np.float32)
        except IndexError:
            continue

        rr, cc = skimage.draw.line(
            image_coords[edge[0]][1], image_coords[edge[0]][0], image_coords[edge[1]][1], image_coords[edge[1]][0]
        )

        if not (rr_length := len(rr)):
            continue

        color_steps = interpolate_func(color0, color1, rr_length)
        image[rr, cc] = color_steps

    return image


def clean_image(image: Image, min_value: int = 0, max_value: int = 255, *, inplace: bool = False) -> Image:
    """
    Clean up NaN values in an image and clip values to a range of min-max.

    This function replaces NaN and infinity values in the provided image.
    It ensures that all values are within the valid range.

    :param image: An image which to cleanup.
    :param min_value: The minimum value of the valid range.
    :param max_value: The maximum value of the valid range.
    :param inplace: Whether to modify the image in place or not. Keyword only argument.
    :return: A cleaned up image with values clipped to the valid range.
    """
    # Avoid side effects on the input image
    if not inplace:
        image = image.copy()
    # Replace NaN values with zero
    image = np.nan_to_num(image, nan=min_value)

    # Replace infinity values with the maximum finite value in the array
    image = np.where(np.isinf(image), np.nanmax(image[np.isfinite(image)]), image)

    # Clip values to be within the range
    image = np.clip(image, min_value, max_value)

    return image


def rasterize(
    image: Image,
    edges: Edges,
    image_coords: PixelCoord,
    colors: Color,
    interpolate_func: Callable[[Color, Color, int], Color] = interpolate_values,
    fill_func: Callable[[Image], Image] = lerp_nans_horizontally,
    *,
    inplace: bool = False,
) -> Image:
    """
    Draw lines with color interpolation and fill NaN values in an image.

    :image: An image to draw lines on and fill NaN values in.
    :param edges: Index pairs into the image_coords and colors arrays for starts and ends of lines.
    :param image_coords: Texture coordinates for the lines' starts and ends.
    :param colors: Array of colors for the starts and ends of lines.
    :param interpolate_func: Function to interpolate color values between the start and end of a line. Default Lerp.
    :param fill_func: Function to fill values(default works with NaN). Default lerp horizontally.
    :param inplace: Whether to modify the image in place or not. Keyword only argument.
    :return: Image with interpolated lines and filled values.
    """
    # Check for empty inputs and return the input image if one of the parameters are not valid
    if edges.size == 0 or image_coords.size == 0 or colors.size == 0:
        return clean_image(image, inplace=inplace)

    try:
        unique_indices = np.unique(edges.flatten())
        # Failing early before proceeding with the code because draw line loops over the indices
        image_coords[unique_indices]
        colors[unique_indices]
    except IndexError as error:
        max_edge_index = unique_indices.max()
        msg = (
            "An edge index is out of bounds. "
            f"Max edge index: {max_edge_index}, "
            f"Image coords shape: {image_coords.shape}, "
            f"Vertex colors shape: {colors.shape}."
        )
        raise IndexError(msg) from error

    if not inplace:
        image = image.copy()
    image = draw_lines(image, edges, image_coords, colors, interpolate_func)
    image = fill_func(image)

    image = clean_image(image, inplace=True)

    return image

import typing

import numpy as np
import skimage

from readyplayerme.meshops.types import Color, Edges, Image, PixelCoord


def interpolate_points(start: Color, end: Color, num_steps: int) -> Color:
    """
    Return an array with interpolated values between start and end.

    The array includes start and end values and is of length num_steps+2, with the first element being start,
    and the last being end, and in between elements are linearly interpolated between these values.

    :param start: The starting value(s) for interpolation.
    :param end: The ending value(s) for interpolation.
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


def lerp_nans_horizontally(image_array: Image) -> Image:
    """
    Linearly interpolates over NaN values in a 2D array, horizontally.

    This function applies linear interpolation across each row of the array, filling NaN values based on adjacent
    non-NaN elements in the same row. Edge NaNs in a row are filled with the nearest valid values in that row.

    :param image_array: A 2D NumPy array to interpolate over. Each row of the array is processed separately.
    :return: A 2D NumPy array with NaN values in each row replaced by interpolated values.
    """
    if image_array.ndim == 1:
        image_array = image_array[np.newaxis, :]

    return np.apply_along_axis(interpolate_segment, 1, image_array)


def lerp_nans_vertically(image_array: Image) -> Image:
    """Linearly interpolates over NaN values in a 2D array, vertically.

    :param image_array: The array to interpolate over.
    :return: The interpolated array.
    """
    if image_array.ndim == 1:
        image_array = image_array[:, np.newaxis]

    return np.apply_along_axis(interpolate_segment, 0, image_array)


def create_nan_image_array(width: int, height: int) -> Image:
    """
    Create an image array filled with NaN values.

    :param width: Width of the image in pixels.
    :param height: Height of the image in pixels.
    :return: A NumPy array of shape (height, width, 3) filled with NaN values.
    """
    try:
        return np.full((height, width, 3), np.nan, dtype=np.float32)
    except ValueError as error:
        msg = "Width and height must be positive integers"
        raise ValueError(msg) from error


def draw_lines(
    image_array: Image,
    edges: Edges,
    image_coords: PixelCoord,
    colors: Color,
    interpolate_func: typing.Callable[[Color, Color, int], Color],
) -> Image:
    """
    Draw lines with color interpolation on an image array.

    :param image_array: The image array to draw lines on.
    :param edges: List of tuples representing the start and end indices of the edges
    from the image_coords and colors array.
    :param image_coords: Texture coordinates for the edges.
    :param colors: Array of colors.
    :param interpolate_func: Function to interpolate colors.
    :return: Image array with interpolated lines.
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
        image_array[rr, cc] = color_steps

    return image_array


def rasterize(
    width: int,
    height: int,
    edges: Edges,
    image_coords: PixelCoord,
    colors: Color,
    interpolate_func: typing.Callable[[Color, Color, int], Color],
    fill_func: typing.Callable[[Image], Image],
) -> Image:
    """
    Draw lines with color interpolation and fill NaN values in an image array.

    :param width: Width of the image in pixels.
    :param height: Height of the image in pixels.
    :param edges: List of tuples representing the start and end points of the edges.
    :param image_coords: Texture coordinates for the edges.
    :param colors: Array of colors for the vertices.
    :param interpolate_func: Function to interpolate values(colors, normals, etc..) between the 2 points.
    :param fill_func: Function to fill NaN values.
    :return: Image array with interpolated lines and filled values.
    """
    # Check for empty inputs
    if edges.size == 0 or image_coords.size == 0 or colors.size == 0:
        return np.zeros((height, width, 3), dtype=np.float32)

    try:
        unique_indices = np.unique(edges.flatten())
        # Attempt to access the maximum index in image_coords and color
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

    image_array = create_nan_image_array(width, height)
    image_array = draw_lines(image_array, edges, image_coords, colors, interpolate_func)
    image_array = fill_func(image_array)

    # Replace infinities and ensure values are in the valid range before type conversion
    image_array = np.nan_to_num(image_array, nan=0)
    image_array = np.where(np.isinf(image_array), np.nanmax(image_array[np.isfinite(image_array)]), image_array)
    image_array = np.clip(image_array, 0, 255)

    return image_array

import typing

import numpy as np
import skimage

from readyplayerme.meshops.types import Color, Edges, Image, PixelCoord


def vectorized_interpolate(start: Color, end: Color, num_steps: int) -> Color:
    """
    Vectorized custom interpolator using NumPy.

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


def lerp_nans_vertically(image_array: Image) -> Image:
    """Linearly interpolates over NaN values in a 2D array, vertically.

    :param image_array: The array to interpolate over.
    :return: The interpolated array.
    """
    # If the input is 1D, make it 2D so we can iterate over columns.
    if image_array.ndim == 1:
        image_array = image_array[:, np.newaxis]

    nan_mask = np.isnan(image_array)
    for col in range(image_array.shape[1]):
        col_data = image_array[:, col]
        nan_indices = np.where(nan_mask[:, col])[0]
        valid_indices = np.where(~nan_mask[:, col])[0]

        # Check if there are valid points to interpolate from
        if len(valid_indices) == 0 or len(nan_indices) == 0:
            continue  # No valid data in this column to interpolate from

        image_array[nan_indices, col] = np.interp(
            nan_indices, valid_indices, col_data[valid_indices], left=np.nan, right=np.nan
        )
    return image_array


def lerp_nans_horizontally(image_array: Image) -> Image:
    """Linearly interpolates over NaN values in a 2D array, vertically.

    :param image_array: The array to interpolate over.
    :return: The interpolated array.
    """
    # If the input is 1D, make it 2D so we can iterate over rows.
    if image_array.ndim == 1:
        image_array = image_array[np.newaxis, :]

    nan_mask = np.isnan(image_array)
    for row in range(image_array.shape[0]):
        row_data = image_array[row, :]
        nan_indices = np.where(nan_mask[row, :])[0]
        valid_indices = np.where(~nan_mask[row, :])[0]

        # Check if there are valid points to interpolate from
        if len(valid_indices) == 0 or len(nan_indices) == 0:
            continue  # No valid data in this row to interpolate from

        image_array[row, nan_indices] = np.interp(
            nan_indices, valid_indices, row_data[valid_indices], left=np.nan, right=np.nan
        )
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
    :param interpolate_func: Function to interpolate colors.
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

    # TODO make a function out of this for creating a Nan_image_array()
    try:
        # Perform an operation that naturally raises an exception for invalid dimensions
        np.empty((height, width, 3))
    except ValueError as error:
        msg = "Width and height must be positive integers"
        raise ValueError(msg) from error
    image_array = np.full((height, width, 3), np.nan, dtype=np.float32)

    # TODO make a function draw_lines(edge,colors,image_coords) (have a method and interpolator injection)
    for edge in edges:
        color0, color1 = colors[edge].astype(np.float32)

        rr, cc = skimage.draw.line(
            image_coords[edge[0]][1], image_coords[edge[0]][0], image_coords[edge[1]][1], image_coords[edge[1]][0]
        )

        if len(rr) == 0:
            continue

        color_steps = interpolate_func(color0, color1, len(rr))
        image_array[rr, cc] = color_steps

    # TODO make a function
    for channel in range(image_array.shape[2]):
        image_array[:, :, channel] = fill_func(image_array[:, :, channel])
    # Replace infinities and ensure values are in the valid range before type conversion
    image_array = np.nan_to_num(image_array, nan=0)
    image_array = np.where(np.isinf(image_array), np.nanmax(image_array[np.isfinite(image_array)]), image_array)
    image_array = np.clip(image_array, 0, 255)

    return image_array

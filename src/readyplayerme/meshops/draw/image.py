"""Module for dealing with image manipulation."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import skimage
from scipy.ndimage import gaussian_filter, maximum_filter

from readyplayerme.meshops import mesh as mops
from readyplayerme.meshops.draw.color import (
    blend_colors,
    get_color_array_color_mode,
    get_image_color_mode,
    interpolate_values,
)
from readyplayerme.meshops.types import Color, ColorMode, Edges, Image, PixelCoord


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


def clip_image(image: Image, min_value: int = 0, max_value: int = 255, *, inplace: bool = False) -> Image:
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


def blend_images(image1: Image, image2: Image, mask: Image) -> Image:
    """
    Blend two images using a mask.

    This function performs a blending operation on two images using a mask. The mask determines the blending ratio
    at each pixel. The blending is done via a vectorized operation for efficiency.

    :param image1: The first image to blend.
    :param image2: The second image to blend. Must be the same shape as the first image.
    :param mask: The blending mask. Must be the same shape as the images.
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
    """
    Linearly interpolates over NaN values in a 2D array, vertically.

    This function applies linear interpolation across each column of the array, filling NaN values based on adjacent
    non-NaN elements in the same column. Edge NaNs in a column are filled with the nearest valid values in that column.

    :param image: The array to interpolate over.
    :return: The interpolated array.
    """
    if image.ndim == 1:
        image = image[:, np.newaxis]

    return np.apply_along_axis(interpolate_segment, 0, image)


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
    if get_color_array_color_mode(colors) == ColorMode.GRAYSCALE:
        colors = colors[:, np.newaxis]
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


def create_mask(width: int, height: int, triangle_coords: npt.NDArray[np.int16]) -> Image:
    """Create a binary mask image from polygons.

    We use individual triangles to draw the mask, as it's not easy to draw polygons with genus > 0, i.e. with holes.

    :param width: Width of the mask image in pixels.
    :param height: Height of the mask image in pixels.
    :param triangle_coords: The triangles to draw in the mask. Shape (t, 3, 2), t triangles, 3 pairs of 2D coordinates.
    Coordinates are in image space.
    :return: A grayscale image mask as uint8 with white (255) being the masked areas.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for triangle in triangle_coords:
        rr, cc = skimage.draw.polygon(triangle[:, 1], triangle[:, 0])
        mask[rr, cc] = 255
    return mask


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
        return clip_image(image, inplace=inplace)

    # Check if the image and colors are compatible (both grayscale or both color)
    image_mode = get_image_color_mode(image)
    colors_mode = get_color_array_color_mode(colors)
    if image_mode != colors_mode:
        msg = "Color mode of 'image' and 'colors' must match (both grayscale or both color)."
        raise ValueError(msg)

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

    image = clip_image(image, inplace=True)

    return image


def blend_uv_seams(mesh: mops.Mesh, image: Image) -> Image:
    """Blend colors in the given image at matching UV seams of the given mesh.

    :param mesh: The mesh to use for blending the UV seams in the provided image.
    :param image: The image to blend the UV seams in.
    :return: The blended image as uint8.
    """
    # Group boundary vertices that share the same position to identify UV seams.
    boundary_vertices = mops.get_boundary_vertices(mesh.edges)
    seam_vertices = mops.get_overlapping_vertices(mesh.vertices, boundary_vertices)
    # Sample colors from all UV coordinates and blend only colors of overlapping vertices.
    try:
        pixel_coords = mops.uv_to_image_coords(mesh.uv_coords, image.shape[0], image.shape[1])
    except TypeError as error:
        msg = f"UV coordinates are invalid: {mesh.uv_coords}"
        raise ValueError(msg) from error
    vertex_colors = image[pixel_coords[:, 1], pixel_coords[:, 0]]
    mixed_colors = blend_colors(vertex_colors, seam_vertices)

    # Creating the vertex mask.
    vertex_color_mask = np.zeros(len(mesh.vertices), dtype=float)
    # Set overlapping vertices to white to have a 0-1 mask.
    vertex_color_mask[np.concatenate(seam_vertices).flatten()] = 1.0

    # Rasterize the vertex color and mask for blending.
    image_color_mode = get_image_color_mode(image)
    raster_image = create_nan_image(image.shape[0], image.shape[1], image_color_mode)
    raster_image = rasterize(raster_image, mesh.edges, pixel_coords, mixed_colors, inplace=True)
    image_mask = create_nan_image(image.shape[0], image.shape[1], ColorMode.GRAYSCALE)
    image_mask = rasterize(image_mask, mesh.edges, pixel_coords, vertex_color_mask, inplace=True)

    # Make the mask smoother by applying a gaussian filter and remapping values.
    blurred_mask = gaussian_filter(image_mask, sigma=3)
    blurred_mask = np.power(blurred_mask, 1.5)

    # Blend in average vertex color at the UV seams.
    return blend_images(image, raster_image, blurred_mask).astype(np.uint8)


def get_vertex_attribute_image(
    width: int,
    height: int,
    faces: mops.Faces,
    uvs: mops.UVs,
    attribute: Color,
    padding: int = 4,
) -> Image:
    """Turn a vertex attribute into an image using a uv layout.

    If the attribute is not already a uint8, it's normalized and then mapped to the 8-bit range [0, 255].

    :param width: Width of the image in pixels.
    :param height: Height of the image in pixels.
    :param faces: The faces of the mesh containing vertex indices.
    :param uvs: The UV coordinates for the vertices of the faces.
    :param padding: Padding in pixels to add around the UV shells. Default 4.
    :return: The vertex attribute as an 8-bit image.
    """
    # Sanity checks.
    try:
        num_uvs = len(uvs)
    except TypeError as error:
        msg = f"UV coordinates are invalid: {uvs}."
        raise ValueError(msg) from error
    if (num_vertices := faces.ravel().max() + 1) > num_uvs:
        msg = f"UV coordinates are invalid: Too few UV coordinates. Expected {num_vertices}, got {num_uvs}."
        raise ValueError(msg)
    if len(attribute) != num_uvs:
        msg = f"Attribute length does not match UV coordinates length: {len(attribute)} != {num_uvs}."
        raise ValueError(msg)
    try:
        attr_color_mode = get_color_array_color_mode(attribute)
    except ValueError as error:
        msg = f"Attribute shape is unsupported for image conversion: {attribute.shape}"
        raise ValueError(msg) from error

    if attribute.dtype != np.uint8:
        # Map the normalized attribute to the 8-bit color format.
        colors = np.nan_to_num(
            (attribute - attribute.min(axis=0, keepdims=True)) / np.ptp(attribute, axis=0, keepdims=True)
        )  # Fixme per channel opt, extract to function
        colors = skimage.util.img_as_ubyte(colors)
    else:
        colors = attribute

    # Create an image from the attribute.
    image_coords = mops.uv_to_image_coords(uvs, width, height)
    attribute_img = create_nan_image(width, height, attr_color_mode)
    edges = mops.faces_to_edges(faces)
    attribute_img = rasterize(attribute_img, edges, image_coords, colors, inplace=True)
    # Constrain colored pixels to the UV shells.
    triangle_coords = mops.get_faces_image_coords(faces, uvs, width, height)
    mask = create_mask(width, height, triangle_coords).astype(bool)
    if attr_color_mode in (ColorMode.RGB, ColorMode.RGBA):
        mask = mask[:, :, np.newaxis].repeat(attr_color_mode.value, axis=2)
    attribute_img *= mask
    # Add padding around the UV shells.
    if padding > 0:
        padded = maximum_filter(attribute_img, size=padding, axes=[0, 1])
        attribute_img[~mask] = padded[~mask]

    return attribute_img.astype(np.uint8)


def get_position_map(width: int, height: int, mesh: mops.Mesh, padding: int = 4) -> Image:
    """Get a position map from the given mesh.

    The positions are normalized and then mapped to the 8-bit range [0, 255].

    :param width: Width of the position map in pixels.
    :param height: Height of the position map in pixels.
    :param mesh: The mesh to use for creating the position map.
    :param padding: Padding in pixels to add around the UV shells. Default 4.
    :return: The position map as uint8.
    """
    return get_vertex_attribute_image(width, height, mesh.faces, mesh.uv_coords, mesh.vertices, padding=padding)


def get_obj_space_normal_map(width: int, height: int, mesh: mops.Mesh, padding: int = 4) -> Image:
    """Get an object space normal map from the given mesh.

    The normals are mapped to the 8-bit integer range [0, 255] to be represented as colors.

    :param width: Width of the normal map in pixels.
    :param height: Height of the normal map in pixels.
    :param mesh: The mesh to use for creating the normal map.
    :param padding: Padding in pixels to add around the UV shells. Default 4.
    :return: The object space normal map as uint8.
    """
    if mesh.normals is None:
        msg = "Mesh does not have vertex normals."
        raise ValueError(msg)
    return get_vertex_attribute_image(width, height, mesh.faces, mesh.uv_coords, mesh.normals, padding=padding)

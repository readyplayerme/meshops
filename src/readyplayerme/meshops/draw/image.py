"""Module for dealing with image manipulation."""
import numpy as np
from scipy.ndimage import gaussian_filter

from readyplayerme.meshops import draw
from readyplayerme.meshops import mesh as mops
from readyplayerme.meshops.types import ColorMode, Image


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


def blend_uv_seams(mesh: mops.Mesh, image: Image) -> Image:
    """Blend colors in the given image at matching UV seams of the given mesh.

    :param mesh: The mesh to use for blending the UV seams in the provided image.
    :param image: The image to blend the UV seams in.
    :return: The blended image.
    """
    # Group boundary vertices that share the same position to identify UV seams.
    boundary_vertices = mops.get_boundary_vertices(mesh.edges)
    seam_vertices = mops.get_overlapping_vertices(mesh.vertices, boundary_vertices)
    # Sample colors from all UV coordinates and blend only colors of overlapping vertices.
    pixel_coords = mops.uv_to_image_coords(mesh.uv_coords, image.shape[0], image.shape[1])
    vertex_colors = image[pixel_coords[:, 1], pixel_coords[:, 0]]
    mixed_colors = mops.blend_colors(vertex_colors, seam_vertices)

    # Creating the vertex mask
    vertex_color_mask = np.zeros((len(mesh.vertices), 1), dtype=float)  # Todo mask shape should be 1D
    # Set overlapping vertices to white to have a 0-1 mask
    vertex_color_mask[np.concatenate(seam_vertices).flatten()] = [1.0]

    # Rasterize the vertex color and mask for blending.
    raster_image = draw.create_nan_image(image.shape[0], image.shape[1], ColorMode.RGB)
    raster_image = draw.rasterize(raster_image, mesh.edges, pixel_coords, mixed_colors, inplace=True)
    image_mask = draw.create_nan_image(image.shape[0], image.shape[1], ColorMode.GRAYSCALE)
    image_mask = draw.rasterize(image_mask, mesh.edges, pixel_coords, vertex_color_mask, inplace=True)

    # Make the mask smoother by applying a gaussian filter and remapping values.
    blurred_mask = gaussian_filter(image_mask, sigma=3)
    blurred_mask = np.power(blurred_mask, 1.5)

    # Blend in average vertex color at the UV seams.
    return blend_images(image, raster_image, blurred_mask)

from readyplayerme.meshops.types import Image as IMG_type


def blend_images(image1: IMG_type, image2: IMG_type, mask: IMG_type) -> IMG_type:
    """
    Blend two images using a mask.

    This function performs a blending operation on two images using a mask. The mask determines the blending ratio
    at each pixel. The blending is done via a vectorized operation for efficiency.

    :param image_1: The first image to blend, as a NumPy array.
    :param image_2: The second image to blend, as a NumPy array. Must be the same shape as np_image.
    :param mask: The blending mask, as a NumPy array. Must be the same shape as the images.
    :return: The blended image, as a NumPy array.
    """
    try:
        # Check if the error is due to shape mismatch
        if not (image1.shape == image2.shape):
            msg = "image1 and image2 must have the same shape."
            raise ValueError(msg)
        # Define a constant variable for the expected number of dimensions
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

from readyplayerme.meshops.types import Image as IMG_type


def blend_images(image_1: IMG_type, image_2: IMG_type, mask: IMG_type) -> IMG_type:
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
        # Perform the blending operation using vectorized NumPy operations
        blended_image = (1 - mask) * image_1 + mask * image_2
        return blended_image
    except ValueError as error:
        # Check if the error is due to shape mismatch
        if not (image_1.shape == image_2.shape == mask.shape):
            msg = "All inputs must have the same shape."
            raise ValueError(msg) from error
        else:
            # Re-raise the original exception if it's not a shape mismatch
            raise

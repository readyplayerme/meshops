"""Custom types for meshops."""
from enum import Enum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# trimesh uses int64 and float64 for its arrays.
Indices: TypeAlias = npt.NDArray[np.uint32] | npt.NDArray[np.uint64]  # Shape (i,)
Vertices: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]  # Shape (v, 3)
Edges: TypeAlias = npt.NDArray[np.int32] | npt.NDArray[np.int64]  # Shape (e, 2)
Faces: TypeAlias = npt.NDArray[np.int32] | npt.NDArray[np.int64]  # Shape (f, 3)
IndexGroups: TypeAlias = list[npt.NDArray[np.uint32]]
Color: TypeAlias = npt.NDArray[np.uint8]  # Shape RGBA: (c, 4) | RGB: (c, 3) | Grayscale: (c,)


class ColorMode(Enum):
    """Color modes for images."""

    GRAYSCALE = 0
    RGB = 3
    RGBA = 4


# The Image type is based on numpy arrays for compatibility with skimage. Floats are used to allow NANs,
# which are not supported by uint8, but the range of floats is supposed to be [0, 255] for colors and [0, 1] for masks.
Image: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.float32] | npt.NDArray[np.float64]  # Shape (h, w[, c]) 2D/3D


UVs: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]  # Shape (i, 2)
PixelCoord: TypeAlias = npt.NDArray[np.uint16]  # Shape (i, 2)

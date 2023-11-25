"""Custom types for meshops."""
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

# trimesh uses int64 and float64 for its arrays.
Indices: TypeAlias = npt.NDArray[np.uint32] | npt.NDArray[np.uint64]  # Shape (i,)
Vertices: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]  # Shape (v, 3)
Edges: TypeAlias = npt.NDArray[np.int32] | npt.NDArray[np.int64]  # Shape (e, 2)
Faces: TypeAlias = npt.NDArray[np.int32] | npt.NDArray[np.int64]  # Shape (f, 3)
IndexGroups: TypeAlias = list[npt.NDArray[np.uint32]]
Color: TypeAlias = npt.NDArray[np.uint8]  # Shape RGBA: (c, 4) | RGB: (c, 3) | Grayscale: (c,)

UVs: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]  # Shape (i, 2)
PixelCoord: TypeAlias = npt.NDArray[np.uint16]  # Shape (i, 2)


class Mesh(Protocol):
    """Structural type for a mesh class.

    Any class considered a mesh must structurally be compatible with this protocol.
    """

    vertices: Vertices
    edges: Edges
    faces: Faces

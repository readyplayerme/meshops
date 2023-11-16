"""Custom types for meshops."""
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

# trimesh uses int64 and float64 for its arrays.
Indices: TypeAlias = npt.NDArray[np.int32] | npt.NDArray[np.int64]  # Shape (i,)
Vertices: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]  # Shape (v, 3)
Edges: TypeAlias = npt.NDArray[np.int32] | npt.NDArray[np.int64]  # Shape (e, 2)
Faces: TypeAlias = npt.NDArray[np.int32] | npt.NDArray[np.int64]  # Shape (f, 3)
VariableLengthArrays: TypeAlias = list[npt.NDArray[np.int64]]


class Mesh(Protocol):
    """Structural type for a mesh class.

    Any class considered a mesh must structurally be compatible with this protocol.
    """

    vertices: Vertices
    edges: Edges
    faces: Faces

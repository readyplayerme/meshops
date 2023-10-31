"""Functions to handle mesh data and read it from file."""
from collections.abc import Callable
from pathlib import Path

import trimesh


def read_mesh(filename: str | Path) -> trimesh.Trimesh:
    """
    Load 3D model data from a file into a Mesh representation.

    :param filename: The path to the to be loaded.
    :return: The loaded mesh object.
    """
    reader = get_mesh_reader(filename)
    return reader(filename)


def get_mesh_reader(filename: str | Path) -> Callable[[str | Path], trimesh.Trimesh]:
    """Return a reader function for a given file extension."""
    if (ext := Path(filename).suffix) in [".glb", ".gltf"]:
        return read_glb
    msg = f"Unsupported file format: {ext}"
    raise NotImplementedError(msg)


def read_glb(filename: str | Path) -> trimesh.Trimesh:
    """
    Load 3D model data from a GLB file into a Mesh representation.

    :param filename: The path to the GLB file to be loaded.
    :return: The loaded mesh object.
    """
    return trimesh.load(filename, process=False, force="mesh")

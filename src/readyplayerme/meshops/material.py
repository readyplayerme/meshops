from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from trimesh.visual.material import PBRMaterial

from readyplayerme.meshops.types import Color, Image


@dataclass()
class Material:
    """Structural type for a PBR material class.

    Any class considered a material must structurally be compatible with this class.

    Attributes are named after the GLTF 2.0 spec.
    """

    name: str
    baseColorTexture: Image | None  # noqa: N815
    baseColorFactor: Color  # noqa: N815
    doubleSided: bool  # noqa: N815
    emissiveTexture: Image | None  # noqa: N815
    emissiveFactor: npt.NDArray[np.float64]  # noqa: N815  # Shape (3,)
    metallicRoughnessTexture: Image | None  # noqa: N815  # Can have occlusion in red channel.
    metallicFactor: float  # noqa: N815
    roughnessFactor: float  # noqa: N815
    normalTexture: Image | None  # noqa: N815
    occlusionTexture: Image | None  # noqa: N815

    @staticmethod
    def from_trimesh_material(material: PBRMaterial) -> "Material":
        """Create a Material object from a trimesh Material object."""
        base_color_texture = np.array(material.baseColorTexture) if material.baseColorTexture else None
        emissive_texture = np.array(material.emissiveTexture) if material.emissiveTexture else None
        metallic_roughness_texture = (
            np.array(material.metallicRoughnessTexture) if material.metallicRoughnessTexture else None
        )
        normal_texture = np.array(material.normalTexture) if material.normalTexture else None
        occlusion_texture = np.array(material.occlusionTexture) if material.occlusionTexture else None

        return Material(
            name=material.name,
            baseColorTexture=base_color_texture,
            baseColorFactor=material.baseColorFactor or np.array([255, 255, 255, 255], dtype=np.uint8),
            doubleSided=material.doubleSided or False,
            emissiveTexture=emissive_texture,
            emissiveFactor=material.emissiveFactor or np.array([0.0, 0.0, 0.0], dtype=np.float64),
            metallicRoughnessTexture=metallic_roughness_texture,
            metallicFactor=material.metallicFactor or (0.0 if metallic_roughness_texture is None else 1.0),
            roughnessFactor=material.roughnessFactor or 1.0,
            normalTexture=normal_texture,
            occlusionTexture=occlusion_texture,
        )

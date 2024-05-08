# MeshOps &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/readyplayerme/meshops/blob/main/LICENSE.txt)

A collection of tools for dealing with mesh related data.

## Setup

You'll find setup instruction of this project in the [CONTRIBUTING.md](https://github.com/readyplayerme/meshops/blob/main/CONTRIBUTING.md) file.

## Features

### Position Map

```python
from readyplayerme.meshops.draw.image import get_position_map
```

Position maps encode locations on the surface of a mesh as color values in an image using the UV layout.
Since 8-bit colors are positive values only and capped at 255, the linear transformation of the positions into the color space is lossy and not invertible, meaning that the original positions cannot be recovered from the color values.
However, these maps can be utilized as control signals for various tasks such as texture synthesis and for shader effects.

### Object Space Normal Maps

```python
from readyplayerme.meshops.draw.image import get_obj_space_normal_map
```

Object space normal maps encode the surface normals of a mesh as color values in an image using the UV layout.
Similar to position maps, the conversion from normals to colors is lossy.
They also can be used as control signals for various tasks such as texture synthesis.

### Any Vertex Attribute to Image

```python
from readyplayerme.meshops.draw.image import get_vertex_attribute_image
```

This function allows you to convert any vertex attribute of a mesh that can be represented as a color to an image.

### UV Island Mask

```python
from readyplayerme.meshops.draw.image import create_mask
```

This function creates a black and white mask image from the mesh's UV layout.

### UV Seams Transitioning

```python
from readyplayerme.meshops.draw.image import blend_uv_seams
```

UV seams are splits in a triangle mesh that, however, is supposed to represent a continuous surface across these splits.
These splits are necessary to allow the mesh to flatten out cleanly in UV space and have as little distortion in the texture projection as possible.
The UV seams are splits between UV islands for which the projection onto the mesh should appear as seamless as possible.

#### Goal

- identify UV splits in a mesh
- mitigate the mismatch between image content when transitioning over a UV split
